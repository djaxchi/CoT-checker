"""
Download Qwen2.5-1.5B model weights and PRM800K dataset to $SCRATCH/hf_cache.
Run this on the TamIA login node (internet access required).

PRM800K is cloned from https://github.com/openai/prm800k (requires git-lfs).
The repo has four phase files; this script concatenates them into train.jsonl
and test.jsonl with the nested schema: question.problem / label.steps.

Usage:
    python scripts/download_assets_tamia.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRATCH = os.environ["SCRATCH"]
HF_CACHE = f"{SCRATCH}/hf_cache"
os.environ["HF_HOME"] = HF_CACHE

from huggingface_hub import snapshot_download

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
#    Source: https://github.com/openai/prm800k  (git-lfs required)
# ---------------------------------------------------------------------------
print("=== Downloading PRM800K (original OpenAI GitHub format) ===")
out_dir = Path(f"{SCRATCH}/cot_mech/raw/prm800k")
out_dir.mkdir(parents=True, exist_ok=True)

# Check if both files already exist with the right schema
def _check_schema(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path) as f:
            line = f.readline().strip()
        if not line:
            return False
        row = json.loads(line)
        return isinstance(row.get("question"), dict)
    except (json.JSONDecodeError, OSError):
        return False

train_ok = _check_schema(out_dir / "train.jsonl")
test_ok = _check_schema(out_dir / "test.jsonl")

if train_ok and test_ok:
    print("train.jsonl and test.jsonl already exist with correct schema, skipping clone.")
else:
    # Check git-lfs is available
    if subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
        sys.exit(
            "git-lfs not found. Install it first:\n"
            "  module load git-lfs  OR  conda install git-lfs"
        )

    # Enable git-lfs for the current user (idempotent)
    subprocess.run(["git", "lfs", "install"], check=True)

    repo_tmp = Path(f"{SCRATCH}/cot_mech/raw/prm800k_repo")
    if repo_tmp.exists():
        print(f"Repo already cloned at {repo_tmp}, ensuring LFS objects are present...")
        # In case the previous clone skipped LFS, install and pull inside the repo
        subprocess.run(["git", "lfs", "install"], cwd=str(repo_tmp), check=True)
        result = subprocess.run(["git", "lfs", "pull"], cwd=str(repo_tmp))
        if result.returncode != 0:
            sys.exit("git lfs pull failed. Check your git-lfs installation and network access.")
    else:
        print("Cloning openai/prm800k from GitHub (this downloads ~1 GB via git-lfs)...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/openai/prm800k.git", str(repo_tmp)],
            check=True,
        )

    data_dir = repo_tmp / "prm800k" / "data"

    # Phase file layout: phase1 + phase2, each with train/test
    phase_map = {
        "train": ["phase1_train.jsonl", "phase2_train.jsonl"],
        "test":  ["phase1_test.jsonl",  "phase2_test.jsonl"],
    }

    for split, phase_files in phase_map.items():
        out = out_dir / f"{split}.jsonl"
        if _check_schema(out):
            print(f"{split}: already correct, skipping.")
            continue
        if out.exists():
            print(f"{split}: wrong schema, rebuilding.")

        print(f"Building {split}.jsonl from {phase_files} ...")
        n_rows = 0
        with open(out, "w") as fout:
            for fname in phase_files:
                src = data_dir / fname
                if not src.exists():
                    sys.exit(f"Expected file not found after git-lfs pull: {src}")
                with open(src) as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            fout.write(line + "\n")
                            n_rows += 1

        # Validate first row schema
        with open(out) as f:
            first = json.loads(f.readline())
        if not isinstance(first.get("question"), dict):
            sys.exit(
                f"Unexpected schema in {out}: keys={list(first.keys())}\n"
                "The GitHub repo may have changed format."
            )
        print(f"{split}: {n_rows} rows -> {out}")

print("\nAll done.")
