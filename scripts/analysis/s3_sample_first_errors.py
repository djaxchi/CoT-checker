"""Sample first-error steps stratified by subset x detected, with full context.

Produces a review JSONL (one first-error step per line) carrying everything the
LLM judge and a human verifier need: the problem, the prior reasoning steps, the
first-error step itself, plus probe metadata (score, whether the probe detected
it). Detection is held out of the judge prompt downstream so the failure-mode
label is unbiased; it is kept here so we can compare detected vs missed.

Output: results/s3_first_error/first_error_sample.jsonl

Usage:
    python scripts/analysis/s3_sample_first_errors.py            # ~25 per cell
    python scripts/analysis/s3_sample_first_errors.py --per_cell 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.processbench_probe_data import DEFAULT_RUN_DIR, SUBSETS, load_all


def _pb_steps_by_id(subset: str) -> dict[str, dict]:
    from datasets import load_dataset

    ds = load_dataset("Qwen/ProcessBench", split=subset)
    return {r["id"]: {"problem": r["problem"], "steps": r["steps"]} for r in ds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--out", type=Path,
                    default=Path("results/s3_first_error/first_error_sample.jsonl"))
    ap.add_argument("--per_cell", type=int, default=25,
                    help="target samples per (subset x detected) cell")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = load_all(args.run_dir, with_text=False)  # text re-joined from raw below
    fe = d.is_first_error & ~d.skipped
    idx_fe = np.where(fe)[0]
    detected = (d.pred_first_error == d.gold_first_error)

    pb_text: dict[str, dict[str, dict]] = {s: _pb_steps_by_id(s) for s in SUBSETS}

    records = []
    for sub in SUBSETS:
        for det in (True, False):
            cell = idx_fe[(d.subset[idx_fe] == sub) & (detected[idx_fe] == det)]
            take = min(args.per_cell, len(cell))
            chosen = rng.choice(cell, size=take, replace=False) if take else []
            for i in chosen:
                tid = d.trace_id[i]
                k = int(d.step_idx[i])
                row = pb_text[sub].get(tid, {"problem": "", "steps": []})
                records.append({
                    "sample_id": f"{tid}#s{k}",
                    "subset": sub,
                    "trace_id": tid,
                    "step_idx": k,
                    "n_steps": int(d.n_steps[i]),
                    "gold_first_error": int(d.gold_first_error[i]),
                    "detected": bool(det),
                    "probe_score": round(float(d.score[i]), 4),
                    "problem": row["problem"],
                    "prior_steps": row["steps"][:k],
                    "error_step": row["steps"][k] if k < len(row["steps"]) else "",
                })

    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # quick stratification report
    print(f"[sample] wrote {len(records)} steps -> {args.out}")
    for sub in SUBSETS:
        nd = sum(1 for r in records if r["subset"] == sub and r["detected"])
        nm = sum(1 for r in records if r["subset"] == sub and not r["detected"])
        print(f"  {sub:14s} detected={nd:3d}  missed={nm:3d}")


if __name__ == "__main__":
    main()
