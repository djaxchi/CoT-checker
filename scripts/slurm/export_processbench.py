#!/usr/bin/env python3
"""Export ProcessBench splits to plain JSONL files (run on the login node).

Compute nodes on Tamia have no internet access, and HF offline mode flags
interfere with dataset loading. Running this once on the login node produces
self-contained JSONL files that encode_processbench.py can load via --data-file
without any Hub calls.

Usage (login node, after activating venv and setting HF_HOME):
    export HF_HOME=$SCRATCH/hf_cache
    python scripts/slurm/export_processbench.py --out-dir $SCRATCH/cot-checker/processbench
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True,
                   help="Directory to write JSONL files into")
    p.add_argument("--splits", nargs="+", default=["gsm8k", "math"],
                   choices=["gsm8k", "math", "olympiadbench", "omnimath"])
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"Loading split '{split}'...")
        ds = load_dataset("Qwen/ProcessBench", split=split)
        path = out / f"processbench_{split}.jsonl"
        with open(path, "w") as f:
            for row in ds:
                f.write(json.dumps(row) + "\n")
        print(f"  {len(ds)} rows -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
