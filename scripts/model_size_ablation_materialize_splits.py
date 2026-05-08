#!/usr/bin/env python3
"""Materialize Math-Shepherd and ProcessBench split IDs before extraction.

Runs once, no GPU required.  All four model extraction jobs load from the
resulting JSON file so they encode exactly the same steps -- regardless of
streaming order, per-job parse failures, or other sources of divergence.

Output: {output_dir}/splits.json
    {
      "meta": {"ms_per_class": 10000, "seed": 42, ...},
      "ms_train": [[solution_id, step_pos, label], ...],   // 2 * ms_per_class rows
      "ms_eval":  [[solution_id, step_pos, label], ...],   // 2 * ms_per_class rows
    }
    (ProcessBench uses all records in dataset order; no sampling needed.)

Usage:
    python scripts/model_size_ablation_materialize_splits.py \\
        --output-dir $SCRATCH/cot-checker/ms_ablation \\
        [--ms-data-file $STORE/data/math_shepherd_gsm8k.jsonl] \\
        [--ms-per-class 10000] \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ---- Math-Shepherd parser (identical to model_size_ablation_extract.py) ----

def _parse_ms_entry(entry: dict, solution_id: int) -> list[dict]:
    label_str = entry["label"]
    q_match   = re.search(r"^(.*?)(?=Step 1:)", label_str, re.DOTALL)
    question  = q_match.group(1).strip() if q_match else ""

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


def load_ms_records(data_file: str | None, cache_dir: str | None) -> list[dict]:
    """Load and parse all GSM8K Math-Shepherd records."""
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

    all_records: list[dict] = []
    solution_counter = 0
    for entry in tqdm(entries, desc="Parsing"):
        if entry.get("task") != "GSM8K":
            continue
        all_records.extend(_parse_ms_entry(entry, solution_counter))
        solution_counter += 1

    print(f"  Parsed {len(all_records):,} steps from {solution_counter:,} solutions")
    return all_records


def sample_splits(
    all_records: list[dict],
    n_per_class: int,
    seed: int,
) -> tuple[list[list], list[list]]:
    """Return (train_ids, eval_ids) as [[solution_id, step_pos, label], ...]."""
    correct   = [r for r in all_records if r["label"] == 1]
    incorrect = [r for r in all_records if r["label"] == 0]

    needed = 2 * n_per_class  # train + eval
    print(f"  Correct   : {len(correct):,}  (need {needed:,})")
    print(f"  Incorrect : {len(incorrect):,}  (need {needed:,})")

    if len(correct) < needed or len(incorrect) < needed:
        raise ValueError(
            f"Insufficient data: {len(correct)} correct, {len(incorrect)} incorrect; "
            f"need {needed} of each class."
        )

    rng     = np.random.default_rng(seed)
    cor_idx = rng.choice(len(correct),   size=needed, replace=False)
    inc_idx = rng.choice(len(incorrect), size=needed, replace=False)

    cor_pool = [correct[i]   for i in cor_idx]
    inc_pool = [incorrect[i] for i in inc_idx]

    # First n_per_class of each class → train; next n_per_class → eval
    train_records = cor_pool[:n_per_class]  + inc_pool[:n_per_class]
    eval_records  = cor_pool[n_per_class:]  + inc_pool[n_per_class:]

    rng.shuffle(train_records)
    rng.shuffle(eval_records)

    def to_ids(recs: list[dict]) -> list[list]:
        return [[r["solution_id"], r["step_pos"], r["label"]] for r in recs]

    return to_ids(train_records), to_ids(eval_records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize Math-Shepherd split IDs")
    p.add_argument("--output-dir",   required=True,
                   help="Directory to write splits.json (e.g. $SCRATCH/cot-checker/ms_ablation)")
    p.add_argument("--ms-data-file", default=None,
                   help="Local Math-Shepherd JSONL.  Omit to load from HF cache.")
    p.add_argument("--cache-dir",    default=None)
    p.add_argument("--ms-per-class", type=int, default=10_000,
                   help="Steps per class per split.  Train gets ms_per_class of each class, "
                        "eval gets the same again (non-overlapping).")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--force",        action="store_true",
                   help="Overwrite existing splits.json")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = out_dir / "splits.json"

    if splits_path.exists() and not args.force:
        print(f"splits.json already exists at {splits_path}")
        print("Use --force to regenerate.  Loading and verifying …")
        with open(splits_path) as f:
            splits = json.load(f)
        print(f"  ms_train : {len(splits['ms_train']):,} records")
        print(f"  ms_eval  : {len(splits['ms_eval']):,} records")
        print(f"  meta     : {splits['meta']}")
        return

    all_records = load_ms_records(args.ms_data_file, args.cache_dir)
    train_ids, eval_ids = sample_splits(all_records, args.ms_per_class, args.seed)

    n_train_correct   = sum(1 for r in train_ids if r[2] == 1)
    n_train_incorrect = sum(1 for r in train_ids if r[2] == 0)
    n_eval_correct    = sum(1 for r in eval_ids  if r[2] == 1)
    n_eval_incorrect  = sum(1 for r in eval_ids  if r[2] == 0)

    splits = {
        "meta": {
            "ms_per_class":      args.ms_per_class,
            "seed":              args.seed,
            "n_train":           len(train_ids),
            "n_eval":            len(eval_ids),
            "n_train_correct":   n_train_correct,
            "n_train_incorrect": n_train_incorrect,
            "n_eval_correct":    n_eval_correct,
            "n_eval_incorrect":  n_eval_incorrect,
        },
        "ms_train": train_ids,
        "ms_eval":  eval_ids,
    }

    with open(splits_path, "w") as f:
        json.dump(splits, f)

    print(f"\nSaved {splits_path}")
    print(f"  ms_train : {len(train_ids):,}  ({n_train_correct:,} correct + {n_train_incorrect:,} incorrect)")
    print(f"  ms_eval  : {len(eval_ids):,}   ({n_eval_correct:,} correct + {n_eval_incorrect:,} incorrect)")


if __name__ == "__main__":
    main()
