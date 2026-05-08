#!/usr/bin/env python3
"""Model-size ablation: extract dense hidden states from a bare Qwen2.5 backbone.

No SSAE -- loads the base LM directly and records the last-layer last-token
hidden state for each reasoning step.

Hidden states are saved RAW (not L2-normalised).  Normalisation is applied
at probe-training time via --repr {raw,l2} in model_size_ablation_probe.py,
so both variants can be explored without re-running extraction.

Which examples to encode is controlled by --split-file (output of
model_size_ablation_materialize_splits.py).  All four model sizes must use
the same splits.json to guarantee identical example IDs.

Output files written to --output-dir:
  ms_train_{tag}.npz
      hidden_states  float16  (N, d)   raw, not normalised
      labels         int8     (N,)     1=correct 0=incorrect
      solution_ids   int32    (N,)

  ms_eval_{tag}.npz
      (same keys)

  pb_{tag}.npz
      hidden_states   float16  (N, d)
      step_labels     int8     (N,)
      solution_ids    int32    (N,)
      step_positions  int8     (N,)

Usage:
    python scripts/model_size_ablation_extract.py \\
        --model-id   Qwen/Qwen2.5-7B \\
        --model-tag  7b \\
        --split-file $SCRATCH/cot-checker/ms_ablation/splits.json \\
        --output-dir $SCRATCH/cot-checker/ms_ablation \\
        --device     cuda
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_ms_entry(entry: dict, solution_id: int) -> list[dict]:
    """Parse one Math-Shepherd entry into per-step records (mirrors generate_probe_data.parse_entry)."""
    label_str = entry["label"]
    q_match = re.search(r"^(.*?)(?=Step 1:)", label_str, re.DOTALL)
    question = q_match.group(1).strip() if q_match else ""

    step_blocks = re.findall(
        r"(Step \d+:.*?)\s*([+\-])\s*(?=Step \d+:|$)",
        label_str,
        re.DOTALL,
    )

    records = []
    prior: list[str] = []
    for step_pos, (step_text, sign) in enumerate(step_blocks):
        clean = re.sub(r"<<[^>]*>>", "", step_text).strip()
        context = (question + " " + " ".join(prior)).strip()
        records.append({
            "context":     context,
            "text":        clean,
            "label":       1 if sign == "+" else 0,
            "solution_id": solution_id,
            "step_pos":    step_pos,
        })
        prior.append(clean)
    return records


def parse_pb_entry(row: dict, row_idx: int) -> list[dict]:
    """Parse one ProcessBench row into per-step records (mirrors encode_processbench.build_records)."""
    problem = row["problem"]
    steps   = row["steps"]
    label   = row["label"]   # -1 = all correct, k = first-error index

    n_include = len(steps) if label == -1 else label + 1
    records = []
    prior: list[str] = []
    for i in range(min(n_include, len(steps))):
        context = (problem + " " + " ".join(prior)).strip() if prior else problem
        step_correct = 1 if (label == -1 or i < label) else 0
        records.append({
            "context":     context,
            "text":        steps[i],
            "step_label":  step_correct,
            "step_pos":    i,
            "solution_id": row_idx,
        })
        prior.append(steps[i])
    return records


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def build_token_seqs(
    tokenizer,
    contexts: list[str],
    steps: list[str],
    max_seq_len: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build raw (input_ids, attn_mask) lists with left-truncation."""
    all_ids, all_masks = [], []
    for ctx, step in zip(contexts, steps):
        ctx_ids  = tokenizer.encode(ctx,  add_special_tokens=False)
        step_ids = tokenizer.encode(step, add_special_tokens=False)
        # [context \n step eos]  -- '\n' separates context from step
        nl_id  = tokenizer.encode("\n", add_special_tokens=False)
        eos_id = tokenizer.eos_token_id
        seq = ctx_ids + nl_id + step_ids + [eos_id]
        if len(seq) > max_seq_len:
            overhead = len(nl_id) + len(step_ids) + 1  # nl + step + eos
            keep_ctx = max_seq_len - overhead
            ctx_ids  = ctx_ids[-max(keep_ctx, 0):]
            seq = ctx_ids + nl_id + step_ids + [eos_id]
        all_ids.append(seq)
        all_masks.append([1] * len(seq))
    return all_ids, all_masks


@torch.no_grad()
def encode_batch(
    model,
    tokenizer,
    contexts: list[str],
    steps: list[str],
    device: str,
    max_seq_len: int,
) -> np.ndarray:
    """Return raw last-layer last-token embeddings, shape (B, d), float16.

    No L2 normalisation applied here.  Normalisation is the probe's choice.
    """
    all_ids, all_masks = build_token_seqs(tokenizer, contexts, steps, max_seq_len)
    max_len  = max(len(s) for s in all_ids)
    pad_id   = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    B        = len(all_ids)

    # Left-pad so position [:, -1] is always the last real token.
    input_ids  = torch.full((B, max_len), pad_id,  dtype=torch.long)
    attn_mask  = torch.zeros((B, max_len),           dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(all_ids, all_masks)):
        n = len(ids)
        input_ids[i, max_len - n:] = torch.tensor(ids,  dtype=torch.long)
        attn_mask[i, max_len - n:] = torch.tensor(mask, dtype=torch.long)

    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    out = model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
    h = out.hidden_states[-1][:, -1, :].float()   # (B, d)
    return h.cpu().numpy().astype(np.float16)


def encode_records(
    model,
    tokenizer,
    records: list[dict],
    label_key: str,
    device: str,
    batch_size: int,
    max_seq_len: int,
    desc: str = "Encoding",
) -> dict:
    """Encode a list of records, return dict of arrays."""
    h_list, label_list, sol_list, pos_list = [], [], [], []
    for i in tqdm(range(0, len(records), batch_size), desc=desc):
        batch = records[i : i + batch_size]
        h = encode_batch(
            model, tokenizer,
            [r["context"] for r in batch],
            [r["text"]    for r in batch],
            device, max_seq_len,
        )
        h_list.append(h)
        label_list.extend(r[label_key]    for r in batch)
        sol_list.extend(r["solution_id"]  for r in batch)
        pos_list.extend(r.get("step_pos", r.get("step_positions", 0)) for r in batch)

    return {
        "hidden_states": np.concatenate(h_list, axis=0),
        "labels":        np.array(label_list, dtype=np.int8),
        "solution_ids":  np.array(sol_list,   dtype=np.int32),
        "step_positions": np.array(pos_list,  dtype=np.int8),
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_math_shepherd_records(
    data_file:  str | None,
    cache_dir:  str | None,
    split_file: str,
) -> tuple[list[dict], list[dict]]:
    """Return (train_records, eval_records) using IDs from a pre-materialised split file.

    The split file (produced by model_size_ablation_materialize_splits.py) contains
    explicit (solution_id, step_pos, label) tuples so every model extraction job
    encodes exactly the same examples.
    """
    with open(split_file) as f:
        splits = json.load(f)

    train_keys = {(r[0], r[1]): r[2] for r in splits["ms_train"]}
    eval_keys  = {(r[0], r[1]): r[2] for r in splits["ms_eval"]}
    all_keys   = set(train_keys) | set(eval_keys)
    print(f"  Split file: {split_file}")
    print(f"  IDs to collect: {len(all_keys):,}  "
          f"(train={len(train_keys):,}  eval={len(eval_keys):,})")

    if data_file:
        print(f"Loading Math-Shepherd from local file: {data_file}")
        with open(data_file) as f:
            entries = [json.loads(l) for l in f if l.strip()]
    else:
        from datasets import load_dataset
        print("Loading Math-Shepherd from HF cache …")
        entries = list(load_dataset(
            "peiyi9979/Math-Shepherd",
            split="train",
            cache_dir=cache_dir,
        ))

    # Build record lookup keyed by (solution_id, step_pos)
    lookup: dict[tuple, dict] = {}
    solution_counter = 0
    for entry in tqdm(entries, desc="Parsing Math-Shepherd"):
        if entry.get("task") != "GSM8K":
            continue
        for r in parse_ms_entry(entry, solution_counter):
            key = (r["solution_id"], r["step_pos"])
            if key in all_keys:
                lookup[key] = r
        solution_counter += 1
        if len(lookup) == len(all_keys):
            break   # found everything; no need to read further

    missing = all_keys - set(lookup)
    if missing:
        raise ValueError(
            f"{len(missing)} split IDs not found in the dataset. "
            "This means the dataset changed or the split file was generated from a different snapshot."
        )

    # Reconstruct train / eval in split-file order (deterministic)
    train_recs = [lookup[(r[0], r[1])] for r in splits["ms_train"]]
    eval_recs  = [lookup[(r[0], r[1])] for r in splits["ms_eval"]]
    return train_recs, eval_recs


def load_processbench_records(
    data_file: str | None,
    cache_dir:  str | None,
) -> list[dict]:
    if data_file:
        print(f"Loading ProcessBench from local file: {data_file}")
        import json as _json
        with open(data_file) as f:
            ds = [_json.loads(l) for l in f if l.strip()]
    else:
        from datasets import load_dataset
        print("Loading ProcessBench (GSM8K split)…")
        ds = list(load_dataset(
            "Qwen/ProcessBench",
            split="gsm8k",
            cache_dir=cache_dir,
        ))

    records: list[dict] = []
    for row_idx, row in enumerate(ds):
        for r in parse_pb_entry(row, row_idx):
            records.append({
                "context":     r["context"],
                "text":        r["text"],
                "label":       r["step_label"],
                "solution_id": r["solution_id"],
                "step_pos":    r["step_pos"],
            })
    print(f"  ProcessBench steps: {len(records):,}")
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract dense hidden states for model-size ablation")
    p.add_argument("--model-id",   required=True, help="HuggingFace model ID, e.g. Qwen/Qwen2.5-7B")
    p.add_argument("--model-tag",  required=True, help="Short tag for filenames, e.g. 7b")
    p.add_argument("--split-file", required=True,
                   help="Path to splits.json from model_size_ablation_materialize_splits.py. "
                        "Ensures all model sizes encode the same examples.")
    p.add_argument("--output-dir", required=True, help="Directory for output .npz files")
    p.add_argument("--ms-data-file", default=None,
                   help="Local Math-Shepherd JSONL (one entry per line). Omit to load from HF cache.")
    p.add_argument("--pb-data-file", default=None,
                   help="Local ProcessBench JSONL. Omit to load from HF cache.")
    p.add_argument("--cache-dir",  default=None, help="HuggingFace cache dir")
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--max-seq-len",  type=int, default=1024)
    p.add_argument("--dtype",        default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device",       default="cuda")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # --- Load model ---
    print(f"\nLoading {args.model_id} ({args.dtype}) …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        device_map=args.device,
    )
    model.eval()
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  hidden_size={d}  num_layers={n_layers}")

    # --- Math-Shepherd ---
    print("\n--- Math-Shepherd ---")
    train_recs, eval_recs = load_math_shepherd_records(
        data_file  = args.ms_data_file,
        cache_dir  = args.cache_dir,
        split_file = args.split_file,
    )
    print(f"  Train: {len(train_recs):,}  |  Eval: {len(eval_recs):,}")

    ms_train = encode_records(
        model, tokenizer, train_recs, "label",
        args.device, args.batch_size, args.max_seq_len,
        desc=f"MS train [{args.model_tag}]",
    )
    ms_eval = encode_records(
        model, tokenizer, eval_recs, "label",
        args.device, args.batch_size, args.max_seq_len,
        desc=f"MS eval  [{args.model_tag}]",
    )

    np.savez_compressed(out_dir / f"ms_train_{args.model_tag}.npz", **ms_train)
    np.savez_compressed(out_dir / f"ms_eval_{args.model_tag}.npz",  **ms_eval)
    print(f"  Saved ms_train_{args.model_tag}.npz  shape={ms_train['hidden_states'].shape}")
    print(f"  Saved ms_eval_{args.model_tag}.npz   shape={ms_eval['hidden_states'].shape}")

    # --- ProcessBench ---
    print("\n--- ProcessBench ---")
    pb_recs = load_processbench_records(args.pb_data_file, args.cache_dir)
    pb_data = encode_records(
        model, tokenizer, pb_recs, "label",
        args.device, args.batch_size, args.max_seq_len,
        desc=f"PB       [{args.model_tag}]",
    )
    # Rename labels → step_labels for ProcessBench convention
    pb_out = {
        "hidden_states":  pb_data["hidden_states"],
        "step_labels":    pb_data["labels"],
        "solution_ids":   pb_data["solution_ids"],
        "step_positions": pb_data["step_positions"],
    }
    np.savez_compressed(out_dir / f"pb_{args.model_tag}.npz", **pb_out)
    print(f"  Saved pb_{args.model_tag}.npz  shape={pb_out['hidden_states'].shape}")

    print(f"\nDone. All files written to {out_dir}")


if __name__ == "__main__":
    main()
