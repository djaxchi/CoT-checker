#!/usr/bin/env python3
"""Materialize ProcessBench raw-trace JSONL for one or more subsets from the
cached Qwen/ProcessBench HF dataset (offline). One trace per line, matching the
existing processbench_gsm8k.jsonl schema: {id, generator, problem, steps, label}.

Usage:
    python scripts/build_processbench_subsets_jsonl.py \
      --subsets olympiadbench omnimath \
      --out_dir /scratch/d/dchikhi/cot-checker/processbench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subsets", nargs="+",
                   default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--dataset", default="Qwen/ProcessBench")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    for subset in args.subsets:
        # ProcessBench exposes each subset as a config with a single "test" split.
        ds = load_dataset(args.dataset, subset, split="test")
        out = args.out_dir / f"processbench_{subset}.jsonl"
        n = 0
        with out.open("w", encoding="utf-8") as f:
            for row in ds:
                rec = {
                    "id": row["id"],
                    "generator": row.get("generator", ""),
                    "problem": row["problem"],
                    "steps": row["steps"],
                    "label": int(row["label"]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"[pb_subsets] {subset}: {n} traces -> {out}", flush=True)


if __name__ == "__main__":
    main()
