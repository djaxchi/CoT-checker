#!/usr/bin/env python3
"""Patch an existing dense .npz to add solution_ids and step_positions.

The dense .npz files were generated with an older version of
generate_probe_data.py that did not save solution_ids / step_positions.
This script re-streams Math-Shepherd (no GPU needed, ~3-5 min) to
reconstruct those arrays and patches the file in-place.

Generation parameters for existing cluster files:
    dense_eval_held_out.npz : --offset 0      --max-steps 90000
    dense_train_full.npz    : --offset 90000  --max-steps 360000

Usage:
    python scripts/add_ms_solution_ids.py \\
        --npz $SCRATCH/cot-checker/probe_data/dense_eval_held_out.npz \\
        --offset 0 --max-steps 90000

    python scripts/add_ms_solution_ids.py \\
        --npz $SCRATCH/cot-checker/probe_data/dense_train_full.npz \\
        --offset 90000 --max-steps 360000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Inline parse_entry — avoids loading torch/SSAE when importing
# ---------------------------------------------------------------------------

def _parse_entry(entry: dict, solution_id: int) -> list[dict]:
    label_str = entry["label"]
    q_match = re.search(r"^(.*?)(?=Step 1:)", label_str, re.DOTALL)
    question = q_match.group(1).strip() if q_match else ""
    step_blocks = re.findall(
        r"(Step \d+:.*?)\s*([+\-])\s*(?=Step \d+:|$)",
        label_str,
        re.DOTALL,
    )
    records = []
    prior_steps: list[str] = []
    for step_pos, (step_text, sign) in enumerate(step_blocks):
        clean = re.sub(r"<<[^>]*>>", "", step_text).strip()
        context = (question + " " + " ".join(prior_steps)).strip()
        records.append({
            "context":     context,
            "text":        clean,
            "label":       1 if sign == "+" else 0,
            "solution_id": solution_id,
            "step_pos":    step_pos,
        })
        prior_steps.append(clean)
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Patch dense .npz with solution_ids and step_positions")
    p.add_argument("--npz",       required=True, help="Path to existing dense .npz")
    p.add_argument("--offset",    type=int, default=0,  help="--offset used during generation")
    p.add_argument("--max-steps", type=int, required=True, help="--max-steps used during generation")
    args = p.parse_args()

    npz_path = Path(args.npz)
    d = np.load(npz_path)

    if "solution_ids" in d.files and "step_positions" in d.files:
        print(f"{npz_path.name}: solution_ids already present. Nothing to do.")
        return

    N = len(d["latents"])
    target = args.offset + args.max_steps
    print(f"{npz_path.name}: {N:,} steps — reconstructing solution structure ...")
    print(f"  Streaming Math-Shepherd up to {target:,} records (offset={args.offset}, "
          f"max_steps={args.max_steps}) ...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset("peiyi9979/Math-Shepherd", split="train", streaming=True)

    all_records: list[dict] = []
    solution_counter = 0
    for entry in ds:
        if entry.get("task") != "GSM8K":
            continue
        all_records.extend(_parse_entry(entry, solution_id=solution_counter))
        solution_counter += 1
        if len(all_records) >= target:
            break

    records = all_records[args.offset : args.offset + args.max_steps]

    if len(records) != N:
        raise ValueError(
            f"Size mismatch: .npz has {N} rows but reconstructed {len(records)} records "
            f"(offset={args.offset}, max_steps={args.max_steps}). "
            f"Check that the same parameters were used during generation."
        )

    sol_ids  = np.array([r["solution_id"] for r in records], dtype=np.int32)
    step_pos = np.array([r["step_pos"]    for r in records], dtype=np.int8)

    n_solutions = int(np.unique(sol_ids).size)
    print(f"  Reconstructed: {n_solutions:,} solutions  "
          f"step_pos range: {step_pos.min()}–{step_pos.max()}")

    # Verify label consistency: reconstructed labels should match stored correctness
    if "correctness" in d.files:
        recon_labels = np.array([r["label"] for r in records], dtype=np.int8)
        mismatches = int((recon_labels != d["correctness"]).sum())
        if mismatches > 0:
            pct = mismatches / N * 100
            print(f"  WARNING: {mismatches:,} label mismatches ({pct:.1f}%) — "
                  f"check offset/max-steps parameters.")
        else:
            print(f"  Label sanity check: all {N:,} labels match. ")

    # Build patched file — preserve all existing arrays, add two new ones
    kwargs = {k: d[k] for k in d.files}
    kwargs["solution_ids"]   = sol_ids
    kwargs["step_positions"] = step_pos

    np.savez_compressed(npz_path, **kwargs)
    size_mb = npz_path.stat().st_size / 1e6
    print(f"Patched → {npz_path}  ({size_mb:.1f} MB)  +solution_ids +step_positions")


if __name__ == "__main__":
    main()
