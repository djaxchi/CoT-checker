#!/usr/bin/env python3
"""Download Math-Shepherd from HuggingFace and export GSM8K rows to a local JSONL.

Run from a login node (which has internet access):
    python scripts/tamia/download_math_shepherd.py
"""
import json
import os
from pathlib import Path

from datasets import load_dataset

cache_dir = os.path.join(os.environ["STORE"], "hf_cache")
out_file  = Path(os.environ["SCRATCH"]) / "cot-checker/ms_ablation/math_shepherd_gsm8k.jsonl"
out_file.parent.mkdir(parents=True, exist_ok=True)

print("Downloading peiyi9979/Math-Shepherd …")
ds = load_dataset("peiyi9979/Math-Shepherd", split="train", cache_dir=cache_dir)
print(f"Total entries : {len(ds)}")

gsm8k = [row for row in ds if row.get("task") == "GSM8K"]
print(f"GSM8K entries : {len(gsm8k)}")

with open(out_file, "w") as f:
    for row in gsm8k:
        f.write(json.dumps(row) + "\n")

print(f"Saved -> {out_file}")
