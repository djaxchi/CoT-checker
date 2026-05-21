"""
Download Qwen2.5-1.5B model weights and PRM800K dataset to $SCRATCH/hf_cache.
Run this on the TamIA login node (internet access required).

Usage:
    python scripts/download_assets_tamia.py
"""

import json
import os
from pathlib import Path

SCRATCH = os.environ["SCRATCH"]
HF_CACHE = f"{SCRATCH}/hf_cache"
os.environ["HF_HOME"] = HF_CACHE

from huggingface_hub import snapshot_download
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Qwen2.5-1.5B model weights
# ---------------------------------------------------------------------------
print("=== Downloading Qwen/Qwen2.5-1.5B ===")
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B",
    repo_type="model",
    ignore_patterns=["*.gguf"],
)
print("Model done.\n")

# ---------------------------------------------------------------------------
# 2. PRM800K dataset
# ---------------------------------------------------------------------------
print("=== Downloading PRM800K ===")
out_dir = Path(f"{SCRATCH}/cot_mech/raw/prm800k")
out_dir.mkdir(parents=True, exist_ok=True)

for split in ("train", "test"):
    out = out_dir / f"{split}.jsonl"
    if out.exists():
        print(f"{split}: already exists at {out}, skipping.")
        continue
    ds = load_dataset("trl-lib/prm800k", split=split, trust_remote_code=True)
    with open(out, "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    print(f"{split}: {len(ds)} rows -> {out}")

print("\nAll done.")
