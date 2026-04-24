#!/usr/bin/env python3
"""Pre-build the eval text record cache for scripts/llm_self_judge.py.

Run this on the LOGIN NODE (needs internet to stream Math-Shepherd).
The cache is shared by all LLM self-judge shards so Math-Shepherd is
only parsed once.

Usage:
    python scripts/build_eval_text_cache.py \
        --out $SCRATCH/cot-checker/probe_data/eval_held_out_text.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.llm_self_judge import collect_eval_text_records  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.expandvars("$SCRATCH/cot-checker/probe_data/eval_held_out_text.jsonl"),
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recs, _ = collect_eval_text_records()
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"Cached {len(recs)} records -> {out}")


if __name__ == "__main__":
    main()
