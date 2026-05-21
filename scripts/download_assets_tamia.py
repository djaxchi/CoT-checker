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
# 2. PRM800K dataset (original OpenAI format with per-candidate ratings)
# ---------------------------------------------------------------------------
print("=== Downloading PRM800K (original OpenAI format) ===")
out_dir = Path(f"{SCRATCH}/cot_mech/raw/prm800k")
out_dir.mkdir(parents=True, exist_ok=True)

# The original dataset has nested schema: question.problem / label.steps
# with per-candidate ratings (+1, -1, 0).
for split in ("train", "test"):
    out = out_dir / f"{split}.jsonl"
    if out.exists():
        # Verify it has the right schema before skipping
        with open(out) as f:
            row = json.loads(f.readline())
        if isinstance(row.get("question"), dict):
            print(f"{split}: already exists with correct schema, skipping.")
            continue
        else:
            print(f"{split}: exists but wrong schema ({list(row.keys())}), re-downloading.")

    ds = load_dataset("openai/prm800k", split=split)
    with open(out, "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    print(f"{split}: {len(ds)} rows -> {out}")

print("\nAll done.")
