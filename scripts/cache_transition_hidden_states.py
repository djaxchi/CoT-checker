#!/usr/bin/env python3
"""Cache (h_k, h_next, delta_h) transition pairs from GSM8K-Aug.

Runs a single LLM forward pass per step and stores the raw backbone hidden
states — no SAE bottleneck. The cached .npz is the input to PTB training.

For --positive-only (default for training), only solutions where every step
passes the symbolic arithmetic judge are included. This ensures the PTB learns
the manifold of *correct* transitions.

Output .npz fields:
    h_k        float16  (N, d)   hidden state at step k
    h_next     float16  (N, d)   hidden state at step k+1
    delta_h    float16  (N, d)   h_next - h_k  (precomputed)
    problem_id int32    (N,)
    step_idx   int8     (N,)     0-indexed position within solution
    num_steps  int8     (N,)     total steps in this solution

Usage:
    python scripts/cache_transition_hidden_states.py \\
        --checkpoint $STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt \\
        --data-file  $STORE/data/gsm8k_385K_train.json \\
        --output     $SCRATCH/cot-checker/probe_data/transition_train_positive.npz \\
        --positive-only \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.saes.ssae import SSAE
from src.data.gsm8k_dataset import symbolic_step_judge


def _tokenise_batch(
    tokenizer,
    contexts: list[str],
    steps: list[str],
    sep_token_id: int,
    max_seq_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded (input_ids, attention_mask) for a batch of (context, step) pairs."""
    batch_ids = []
    for ctx, step in zip(contexts, steps):
        ctx_ids  = tokenizer.encode(ctx,  add_special_tokens=False)
        step_ids = tokenizer.encode(step, add_special_tokens=False)
        seq = ctx_ids + [sep_token_id] + step_ids + [tokenizer.eos_token_id]
        if len(seq) > max_seq_len:
            keep = max_seq_len - len(step_ids) - 2
            ctx_ids = ctx_ids[-max(keep, 0):]
            seq = ctx_ids + [sep_token_id] + step_ids + [tokenizer.eos_token_id]
        batch_ids.append(seq)

    max_len = max(len(s) for s in batch_ids)
    pad_id = tokenizer.eos_token_id
    input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(batch_ids):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attn_mask[i, :len(seq)] = 1
    return input_ids, attn_mask


@torch.no_grad()
def _encode_dense_batch(
    model: SSAE,
    tokenizer,
    contexts: list[str],
    steps: list[str],
    sep_token_id: int,
    max_seq_len: int,
    device: str,
) -> np.ndarray:
    """Return raw backbone hidden states (B, d) as float32 numpy."""
    input_ids, attn_mask = _tokenise_batch(
        tokenizer, contexts, steps, sep_token_id, max_seq_len, device
    )
    vecs = model.encode_dense(input_ids, attn_mask)   # (B, 1, d)
    return vecs.squeeze(1).float().cpu().numpy()


def _load_problems(data_file: str) -> list[dict]:
    """Load GSM8K JSONL — each line has {question, answer} or {question, steps}."""
    problems = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _split_answer_steps(answer: str) -> list[str]:
    """Split a GSM8K answer string into reasoning steps."""
    import re
    # Try numbered step format first
    steps = re.split(r"\n(?=Step \d+:)", answer.strip())
    if len(steps) > 1:
        return [s.strip() for s in steps if s.strip()]
    # Fall back to sentence-based split
    steps = [s.strip() for s in answer.split("\n") if s.strip()]
    return steps if steps else [answer.strip()]


def build_transition_pairs(
    model: SSAE,
    tokenizer,
    problems: list[dict],
    sep_token_id: int,
    batch_size: int,
    max_seq_len: int,
    positive_only: bool,
    device: str,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Iterate over problems, build (h_k, h_next) pairs, return stacked arrays."""

    # Collect all (problem_id, step_idx, context, step, step_text_for_judge, all_steps)
    records = []
    for prob_id, prob in enumerate(tqdm(problems, desc="Parsing problems")):
        question = prob.get("question", "")
        # Support both {answer: str} and {steps: [...]} formats
        if "steps" in prob:
            raw_steps = [s if isinstance(s, str) else s.get("text", "") for s in prob["steps"]]
        elif "answer" in prob:
            raw_steps = _split_answer_steps(prob["answer"])
        else:
            continue

        if len(raw_steps) < 2:
            continue

        # Positive-only filter: skip solution if any step fails arithmetic judge
        if positive_only:
            if not all(symbolic_step_judge(s) == 1 for s in raw_steps):
                continue

        prior: list[str] = []
        for step_idx, step_text in enumerate(raw_steps):
            context = (question + " " + " ".join(prior)).strip()
            records.append({
                "problem_id": prob_id,
                "step_idx":   step_idx,
                "num_steps":  len(raw_steps),
                "context":    context,
                "step":       step_text,
            })
            prior.append(step_text)

    if not records:
        raise ValueError("No valid records found. Check --data-file and --positive-only.")

    print(f"Records to encode: {len(records):,}  (problems: {len(problems):,})")

    # Encode all steps
    all_h: list[np.ndarray] = []
    for i in tqdm(range(0, len(records), batch_size), desc="Encoding hidden states"):
        batch = records[i:i + batch_size]
        ctxs  = [r["context"] for r in batch]
        steps = [r["step"]    for r in batch]
        h = _encode_dense_batch(model, tokenizer, ctxs, steps, sep_token_id, max_seq_len, device)
        all_h.append(h)

    H = np.concatenate(all_h, axis=0)   # (N, d)

    # Build transition pairs: for each step k that has a k+1 in the same solution,
    # record (h_k, h_{k+1}). Last step of each solution is excluded.
    h_k_list, h_next_list, prob_ids, step_idxs, num_steps_list = [], [], [], [], []

    i = 0
    while i < len(records):
        r = records[i]
        # Check if next record is the consecutive step in the same solution
        if (i + 1 < len(records)
                and records[i + 1]["problem_id"] == r["problem_id"]
                and records[i + 1]["step_idx"] == r["step_idx"] + 1):
            h_k_list.append(H[i])
            h_next_list.append(H[i + 1])
            prob_ids.append(r["problem_id"])
            step_idxs.append(r["step_idx"])
            num_steps_list.append(r["num_steps"])
        i += 1

    h_k    = np.stack(h_k_list,    axis=0).astype(np.float16)
    h_next = np.stack(h_next_list, axis=0).astype(np.float16)
    delta  = (h_next.astype(np.float32) - h_k.astype(np.float32)).astype(np.float16)

    print(f"Transition pairs: {len(h_k):,}")
    return {
        "h_k":        h_k,
        "h_next":     h_next,
        "delta_h":    delta,
        "problem_id": np.array(prob_ids,       dtype=np.int32),
        "step_idx":   np.array(step_idxs,      dtype=np.int8),
        "num_steps":  np.array(num_steps_list, dtype=np.int8),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache (h_k, h_next, delta_h) transition pairs")
    p.add_argument("--checkpoint",   required=True, help="SSAE .pt checkpoint (for LLM backbone)")
    p.add_argument("--data-file",    required=True, help="GSM8K-Aug JSONL")
    p.add_argument("--output",       required=True, help="Output .npz path")
    p.add_argument("--positive-only",action="store_true",
                   help="Only include solutions where all steps pass symbolic judge")
    p.add_argument("--batch-size",   type=int,  default=32)
    p.add_argument("--max-seq-len",  type=int,  default=256)
    p.add_argument("--max-problems", type=int,  default=None,
                   help="Cap number of problems to process (for debugging)")
    p.add_argument("--shard-idx",    type=int,  default=0,
                   help="0-indexed shard to process (used with --n-shards for parallel runs)")
    p.add_argument("--n-shards",     type=int,  default=1,
                   help="Total number of shards; each GPU runs one shard")
    p.add_argument("--device",       default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed",         type=int,  default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading SSAE backbone from {args.checkpoint} ...")
    model = SSAE.from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    tokenizer = model.tokenizer
    sep_token_id = tokenizer.sep_token_id

    print(f"Loading problems from {args.data_file} ...")
    problems = _load_problems(args.data_file)
    if args.max_problems:
        problems = problems[:args.max_problems]
    print(f"  {len(problems):,} problems loaded")

    # Shard: each GPU processes a strided subset of problems
    if args.n_shards > 1:
        problems = problems[args.shard_idx::args.n_shards]
        print(f"  Shard {args.shard_idx}/{args.n_shards}: {len(problems):,} problems")

    arrays = build_transition_pairs(
        model        = model,
        tokenizer    = tokenizer,
        problems     = problems,
        sep_token_id = sep_token_id,
        batch_size   = args.batch_size,
        max_seq_len  = args.max_seq_len,
        positive_only= args.positive_only,
        device       = args.device,
        seed         = args.seed,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    N = len(arrays["h_k"])
    print(f"\nSaved {N:,} transition pairs → {out}")
    print(f"  h_k shape: {arrays['h_k'].shape}  dtype: {arrays['h_k'].dtype}")


if __name__ == "__main__":
    main()
