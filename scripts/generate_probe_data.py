#!/usr/bin/env python3
"""Step 1 — Generate probe data from Math-Shepherd (step-level labels).

Math-Shepherd (peiyi9979/Math-Shepherd) provides GSM8K problems where each
reasoning step is labeled + (correct) or - (incorrect) via Monte Carlo rollouts:
a step is correct if, starting from that step, a correct final answer is still
reachable.

Pipeline per step:
  1. Parse (question, step_text, step_label) from Math-Shepherd
  2. Build context = question + all prior steps
  3. Encode [context | <sep> | step] with the SSAE encoder → sparse latent h_c
  4. Store (latent, label)

This gives ground-truth step-level labels completely independent of the SSAE,
matching the research goal: can SSAE features predict step correctness?

Output .npz:
  latents     — float16, shape (N, n_latents)
  correctness — int8,    shape (N,)  {0=incorrect, 1=correct}

Usage:
    python scripts/generate_probe_data.py \\
        --checkpoint gsm8k-385k_Qwen2.5-0.5b_spar-10.pt \\
        --output results/probe_data/math_shepherd_1000.npz \\
        --max-steps 1000 \\
        --max-seq-len 2048 \\
        --device cuda
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.saes.ssae import SSAE

# ки is the step delimiter used in Math-Shepherd input field
STEP_DELIM = "\u043a\u0438"


def parse_entry(entry: dict) -> list[dict]:
    """Parse a Math-Shepherd entry into (context, step_text, label) records.

    The `label` field looks like:
      "Question text Step 1: ... +\\nStep 2: ... -\\n..."
    where + means the step is on a path to the correct answer, - means not.
    """
    label_str = entry["label"]

    # Split off the question: everything before the first "Step 1:"
    q_match = re.search(r"^(.*?)(?=Step 1:)", label_str, re.DOTALL)
    question = q_match.group(1).strip() if q_match else ""

    # Extract each "Step N: <text> <+/->" block
    # Steps end with a bare + or - (possibly surrounded by spaces/newlines)
    step_blocks = re.findall(
        r"(Step \d+:.*?)\s*([+\-])\s*(?=Step \d+:|$)",
        label_str,
        re.DOTALL,
    )

    records = []
    prior_steps: list[str] = []

    for step_text, sign in step_blocks:
        # Strip <<expr=result>> calculator annotations (not natural language)
        clean = re.sub(r"<<[^>]*>>", "", step_text).strip()
        context = (question + " " + " ".join(prior_steps)).strip()
        records.append(
            {
                "context": context,
                "text": clean,
                "label": 1 if sign == "+" else 0,
            }
        )
        prior_steps.append(clean)

    return records


def encode_batch(
    model, tokenizer, contexts, steps, device, sep_token_id, max_seq_len=2048
) -> np.ndarray:
    batch_ids = []
    for ctx, step in zip(contexts, steps):
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
        step_ids = tokenizer.encode(step, add_special_tokens=False)
        seq = ctx_ids + [sep_token_id] + step_ids + [tokenizer.eos_token_id]
        # Truncate from the left so the step is always fully preserved.
        # Overhead = sep + eos = 2 tokens; step must fit within max_seq_len.
        if len(seq) > max_seq_len:
            keep = max_seq_len - len(step_ids) - 2  # tokens available for context
            ctx_ids = ctx_ids[-max(keep, 0):]
            seq = ctx_ids + [sep_token_id] + step_ids + [tokenizer.eos_token_id]
        batch_ids.append(seq)

    max_seq = max(len(s) for s in batch_ids)
    pad_id = tokenizer.eos_token_id
    input_ids = torch.full((len(batch_ids), max_seq), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(batch_ids), max_seq), dtype=torch.long, device=device)
    for i, seq in enumerate(batch_ids):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attn_mask[i, : len(seq)] = 1

    with torch.no_grad():
        latents = model.encode(input_ids, attn_mask)  # (B, 1, n_latents)
    return latents.squeeze(1).cpu().float().numpy()  # (B, n_latents)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="results/probe_data/math_shepherd.npz")
    p.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Total number of steps to encode (drawn from GSM8K split)",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N steps (use to avoid overlap with training data)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Max tokens for the full [context|sep|step] sequence. "
                        "Context is truncated from the left so the step is always fully preserved. "
                        "Set to 0 to disable (use the model's full context window).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument(
        "--correct-ratio",
        type=float,
        default=None,
        help="Target fraction of correct steps (e.g. 0.5 for 50/50). "
             "Subsamples the majority class after collection. Default: no rebalancing.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    import random
    args = parse_args()
    random.seed(args.seed)
    device = args.device

    # --- Load SSAE ---
    model = SSAE.from_checkpoint(args.checkpoint, device=device)
    model.eval()
    tokenizer = model.tokenizer
    sep_tok_id = tokenizer.sep_token_id

    # --- Stream Math-Shepherd, filter GSM8K, collect up to max_steps ---
    print("Streaming Math-Shepherd (GSM8K split)…")
    ds = load_dataset("peiyi9979/Math-Shepherd", split="train", streaming=True)

    all_records: list[dict] = []
    for entry in ds:
        if entry.get("task") != "GSM8K":
            continue
        all_records.extend(parse_entry(entry))
        if len(all_records) >= args.offset + args.max_steps:
            break

    all_records = all_records[args.offset : args.offset + args.max_steps]
    pos_raw = sum(r["label"] for r in all_records)
    print(f"Steps collected : {len(all_records)}")
    print(f"Correct (+)     : {pos_raw}  ({pos_raw / len(all_records):.1%})")
    print(f"Incorrect (-)   : {len(all_records) - pos_raw}  ({(len(all_records) - pos_raw) / len(all_records):.1%})")

    # --- Optional rebalancing ---
    if args.correct_ratio is not None:
        correct_recs   = [r for r in all_records if r["label"] == 1]
        incorrect_recs = [r for r in all_records if r["label"] == 0]
        ratio = args.correct_ratio
        # Keep the minority class intact, subsample the majority
        if len(correct_recs) / len(all_records) > ratio:
            # Too many correct — subsample correct
            n_cor = int(len(incorrect_recs) * ratio / (1 - ratio))
            correct_recs = random.sample(correct_recs, min(n_cor, len(correct_recs)))
        else:
            # Too many incorrect — subsample incorrect
            n_inc = int(len(correct_recs) * (1 - ratio) / ratio)
            incorrect_recs = random.sample(incorrect_recs, min(n_inc, len(incorrect_recs)))
        all_records = correct_recs + incorrect_recs
        random.shuffle(all_records)
        pos_raw = sum(r["label"] for r in all_records)
        print(f"\nAfter rebalancing to {ratio:.0%}/{1-ratio:.0%}:")
        print(f"  Total steps : {len(all_records)}")
        print(f"  Correct (+) : {pos_raw}  ({pos_raw / len(all_records):.1%})")
        print(f"  Incorrect (-): {len(all_records) - pos_raw}  ({(len(all_records) - pos_raw) / len(all_records):.1%})")

    total = len(all_records)
    pos = sum(r["label"] for r in all_records)
    print(f"Majority baseline: {max(pos, total - pos) / total:.1%}")

    # --- Encode in batches ---
    all_latents, all_labels = [], []
    bs = args.batch_size
    max_seq = args.max_seq_len if args.max_seq_len > 0 else 10**9

    for i in tqdm(range(0, total, bs), desc="Encoding with SSAE"):
        batch = all_records[i : i + bs]
        ctxs = [r["context"] for r in batch]
        steps = [r["text"] for r in batch]
        labels = [r["label"] for r in batch]
        lats = encode_batch(model, tokenizer, ctxs, steps, device, sep_tok_id, max_seq_len=max_seq)
        all_latents.extend(lats)
        all_labels.extend(labels)

    # --- Save ---
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        latents=np.array(all_latents, dtype=np.float16),
        correctness=np.array(all_labels, dtype=np.int8),
    )
    print(f"\nSaved {total} samples → {out}")
    print(
        f"Positive rate : {pos / total:.3f}  (majority baseline: {max(pos, total - pos) / total:.3f})"
    )
    print("Paper target majority baseline: 0.705")


if __name__ == "__main__":
    main()
