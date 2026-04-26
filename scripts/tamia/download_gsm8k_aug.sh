#!/usr/bin/env bash
# Run on the TamIA login node (internet access required).
# Downloads Miaow-Lab/SSAE-Dataset (gsm8k config) and writes:
#   $STORE/data/gsm8k_385K_train.json   — full train split (~385K problems)
#   $STORE/data/gsm8k_385K_valid.json   — validation split
#
# Usage:
#   bash scripts/tamia/download_gsm8k_aug.sh

set -euo pipefail

STORE="/project/aip-azouaq/$USER"
DATA_DIR="$STORE/data"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

# Use SCRATCH for HF cache — dataset is re-downloadable
export HF_HOME="$SCRATCH/hf_cache"

mkdir -p "$DATA_DIR"

echo "=== Downloading Miaow-Lab/SSAE-Dataset (gsm8k) ==="
echo "Output dir: $DATA_DIR"

python - <<PYEOF
import json, os, sys
from pathlib import Path
from datasets import load_dataset

data_dir = Path(os.environ["DATA_DIR"])

for split, out_name in [("train", "gsm8k_385K_train.json"), ("validation", "gsm8k_385K_valid.json")]:
    out_path = data_dir / out_name
    print(f"\n  [{split}] -> {out_path}")
    ds = load_dataset("Miaow-Lab/SSAE-Dataset", "gsm8k", split=split)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps({"question": row["question"], "answer": row["answer"]}, ensure_ascii=False) + "\n")
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Wrote {len(ds):,} rows  ({size_mb:.1f} MB)")

print("\nDone.")
PYEOF
