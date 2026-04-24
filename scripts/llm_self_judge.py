#!/usr/bin/env python3
"""LLM-as-judge baseline on the same held-out eval set as the SSAE probe.

For each (question, prior_steps, current_step), we ask the LLM directly:
"Will continuing from this step lead to the correct final answer? Yes/No"

Score = softmax([logit(Yes), logit(No)])[0] on the first assistant token.

Alignment with the probe's eval set:
  The probe is evaluated on 50K steps carved from Math-Shepherd GSM8K
  offsets 0-90K using np.random.default_rng(42) (see reorganize_shards.py).
  This script replays the SAME parse + SAME seed-42 sampling to re-derive
  the same 50K rows as text records, so the LLM sees exactly the steps the
  probe saw.

Shards across GPUs: pass --shard-idx / --n-shards to split work.

Usage:
    python scripts/llm_self_judge.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --output $SCRATCH/cot-checker/results/judge_qwen_0p5b_shard0.npz \\
        --shard-idx 0 --n-shards 4 \\
        --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_probe_data import parse_entry  # type: ignore


EVAL_OFFSET = 0
EVAL_NATURAL_STEPS = 90000     # matches tamia_generate_data.sh eval shard size
EVAL_PER_CLASS = 25000         # matches reorganize_shards.py --eval-per-class
EVAL_SAMPLING_SEED = 42        # matches reorganize_shards.py --seed


def collect_eval_text_records():
    """Re-derive the exact 50K eval records (text form) by replaying the pipeline."""
    print("Streaming Math-Shepherd (GSM8K) and parsing to step records...")
    ds = load_dataset("peiyi9979/Math-Shepherd", split="train", streaming=True)
    all_records = []
    for entry in ds:
        if entry.get("task") != "GSM8K":
            continue
        all_records.extend(parse_entry(entry))
        if len(all_records) >= EVAL_OFFSET + EVAL_NATURAL_STEPS:
            break
    all_records = all_records[EVAL_OFFSET : EVAL_OFFSET + EVAL_NATURAL_STEPS]
    print(f"  Parsed {len(all_records):,} natural steps from offset {EVAL_OFFSET}")

    y = np.array([r["label"] for r in all_records], dtype=np.int8)
    cor_idx = np.where(y == 1)[0]
    inc_idx = np.where(y == 0)[0]
    print(f"  Natural: {len(cor_idx):,} correct / {len(inc_idx):,} incorrect")

    # Replay reorganize_shards.py's sampling exactly
    rng = np.random.default_rng(EVAL_SAMPLING_SEED)
    n = EVAL_PER_CLASS
    eval_cor = rng.choice(cor_idx, size=n, replace=False)
    eval_inc = rng.choice(inc_idx, size=n, replace=False)
    eval_idx = np.concatenate([eval_cor, eval_inc])
    rng.shuffle(eval_idx)

    selected = [all_records[int(i)] for i in eval_idx]
    labels = np.array([r["label"] for r in selected], dtype=np.int8)
    print(f"  Eval set: {len(selected):,} steps "
          f"({int((labels==1).sum()):,} correct / {int((labels==0).sum()):,} incorrect)")
    return selected, labels


def build_prompt(question: str, prior_steps_blob: str, current_step: str) -> str:
    """User-visible prompt matching Math-Shepherd label semantics (path viability)."""
    prior_block = prior_steps_blob.strip() if prior_steps_blob.strip() else "(none)"
    return (
        "You are verifying a math solution step-by-step.\n\n"
        f"Question:\n{question}\n\n"
        f"Solution so far:\n{prior_block}\n\n"
        f"Next step:\n{current_step}\n\n"
        "Will continuing from this next step lead to the correct final answer?\n"
        "Answer with only \"Yes\" or \"No\"."
    )


def split_context(rec: dict) -> tuple[str, str]:
    """parse_entry packs (question + prior_steps) into rec['context'].

    The question ends at the first "Step 1:"; everything after is prior steps.
    When there are no prior steps, the context is just the question.
    """
    ctx = rec["context"]
    marker = "Step 1:"
    i = ctx.find(marker)
    if i == -1:
        return ctx.strip(), ""
    return ctx[:i].strip(), ctx[i:].strip()


def find_yesno_token_ids(tokenizer):
    """Find the token id that a clean 'Yes' / 'No' assistant reply begins with.

    We encode variants and collect any that tokenize to a single leading id,
    then treat all of them as equivalent at scoring time. This is safer than
    assuming one canonical token across tokenizer versions.
    """
    yes_variants = ["Yes", " Yes", "yes", " yes", "YES"]
    no_variants = ["No", " No", "no", " no", "NO"]

    def first_ids(variants):
        ids = set()
        for v in variants:
            toks = tokenizer.encode(v, add_special_tokens=False)
            if toks:
                ids.add(toks[0])
        return ids

    yes_ids = first_ids(yes_variants)
    no_ids = first_ids(no_variants)
    assert yes_ids and no_ids, "Could not resolve Yes/No token ids"
    return sorted(yes_ids), sorted(no_ids)


@torch.no_grad()
def score_batch(model, tokenizer, prompts, device, yes_ids, no_ids, max_ctx):
    """Return P(Yes | prompt) per example via softmax over {Yes-ids, No-ids}."""
    # Apply chat template for each prompt
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    enc = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_ctx,
    ).to(device)

    # Forward pass; take last-token logits (for each row, the last non-pad position)
    out = model(**enc)
    logits = out.logits  # (B, T, V)

    if tokenizer.padding_side == "left":
        last_logits = logits[:, -1, :]
    else:
        # Right-padded: pick the last non-pad token per row
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        last_logits = logits[torch.arange(logits.size(0), device=device), seq_lens, :]

    yes_tensor = torch.tensor(yes_ids, device=device, dtype=torch.long)
    no_tensor = torch.tensor(no_ids, device=device, dtype=torch.long)

    # log-sum-exp over each group for numerical stability
    yes_lse = torch.logsumexp(last_logits.index_select(1, yes_tensor), dim=1)
    no_lse = torch.logsumexp(last_logits.index_select(1, no_tensor), dim=1)
    # softmax over two scalars
    p_yes = torch.sigmoid(yes_lse - no_lse)
    return p_yes.float().cpu().numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--output", required=True, help="Per-shard output .npz (p_yes + labels + indices)")
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-ctx", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--cache-dir", default=None, help="HF cache dir (set to project space on HPC)")
    p.add_argument(
        "--records-cache",
        default=None,
        help="Optional .jsonl path to cache the parsed eval records so "
             "re-parsing Math-Shepherd is only done once.",
    )
    return p.parse_args()


def load_or_build_records(cache_path):
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached eval records from {cache_path}...")
        records, labels = [], []
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                records.append(r)
                labels.append(r["label"])
        labels = np.array(labels, dtype=np.int8)
        print(f"  Loaded {len(records):,} records")
        return records, labels

    records, labels = collect_eval_text_records()
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  Cached eval records -> {cache_path}")
    return records, labels


def main():
    args = parse_args()

    print(f"=== LLM self-judge | shard {args.shard_idx}/{args.n_shards} | model={args.model} ===")

    records, labels = load_or_build_records(args.records_cache)

    # Slice shard: stride the eval set so each GPU gets a representative mix.
    idx = np.arange(len(records))
    my_idx = idx[args.shard_idx :: args.n_shards]
    print(f"  Shard size: {len(my_idx):,} of {len(records):,} total")

    print("\nLoading model...")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # last-token logits are easy with left padding

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    yes_ids, no_ids = find_yesno_token_ids(tok)
    print(f"  Yes token ids: {yes_ids}")
    print(f"  No  token ids: {no_ids}")

    print("\nScoring...")
    scores = np.zeros(len(my_idx), dtype=np.float32)
    my_labels = np.zeros(len(my_idx), dtype=np.int8)

    t0 = time.perf_counter()
    bs = args.batch_size
    for start in tqdm(range(0, len(my_idx), bs)):
        batch_idx = my_idx[start : start + bs]
        prompts = []
        for j in batch_idx:
            r = records[int(j)]
            q, prior = split_context(r)
            prompts.append(build_prompt(q, prior, r["text"]))
        batch_scores = score_batch(model, tok, prompts, args.device, yes_ids, no_ids, args.max_ctx)
        scores[start : start + len(batch_idx)] = batch_scores
        my_labels[start : start + len(batch_idx)] = [records[int(j)]["label"] for j in batch_idx]

    elapsed = time.perf_counter() - t0
    print(f"\n  Scored {len(my_idx):,} steps in {elapsed/60:.1f} min "
          f"({elapsed/len(my_idx)*1000:.1f} ms/step)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        p_yes=scores,
        labels=my_labels,
        indices=my_idx.astype(np.int64),
        model=np.array(args.model),
        shard_idx=np.array(args.shard_idx),
        n_shards=np.array(args.n_shards),
    )
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
