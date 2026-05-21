"""
Verify PRM800K schema and copy ProcessBench gsm8k.json from HF cache.
Run on TamIA login node after download_assets_tamia.py.

Usage:
    python scripts/verify_and_copy_assets.py
"""

import json
import os
import shutil
from pathlib import Path

SCRATCH = os.environ["SCRATCH"]

# ---------------------------------------------------------------------------
# 1. PRM800K schema check
# ---------------------------------------------------------------------------
print("=== PRM800K schema check ===")
train_path = Path(f"{SCRATCH}/cot_mech/raw/prm800k/train.jsonl")
if not train_path.exists():
    print("ERROR: train.jsonl not found.")
else:
    with open(train_path) as f:
        row = json.loads(f.readline())
    print("Keys:", list(row.keys()))
    print("question type:", type(row.get("question")).__name__)
    print("label type:", type(row.get("label")).__name__)

    question = row.get("question")
    if isinstance(question, dict):
        print("Schema: NESTED (question.problem / label.steps) -- CORRECT")
    else:
        print("Schema: FLAT (problem at top level)")
        print("WARNING: build script expects nested schema. Check field names.")

# ---------------------------------------------------------------------------
# 2. Copy ProcessBench gsm8k.json
# ---------------------------------------------------------------------------
print("\n=== ProcessBench gsm8k.json ===")
dest = Path(f"{SCRATCH}/cot_mech/raw/processbench/gsm8k.json")
dest.parent.mkdir(parents=True, exist_ok=True)

if dest.exists():
    print(f"Already exists: {dest}")
else:
    candidates = list(Path(f"{SCRATCH}/hf_cache").rglob("gsm8k.json"))
    if candidates:
        src = candidates[0]
        shutil.copy(src, dest)
        print(f"Copied: {src} -> {dest}")
    else:
        print("gsm8k.json not found in HF cache.")
        print("You need to transfer it from your Mac:")
        print(f"  scp .../gsm8k.json dchikhi@tamia.alliancecan.ca:{dest}")

# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------
print("\n=== File summary ===")
for p in [
    f"{SCRATCH}/hf_cache/hub/models--Qwen--Qwen2.5-1.5B",
    f"{SCRATCH}/cot_mech/raw/prm800k/train.jsonl",
    f"{SCRATCH}/cot_mech/raw/prm800k/test.jsonl",
    f"{SCRATCH}/cot_mech/raw/processbench/gsm8k.json",
]:
    path = Path(p)
    if path.exists():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
        print(f"  OK   {p}  ({size / 1e6:.0f} MB)")
    else:
        print(f"  MISS {p}")
