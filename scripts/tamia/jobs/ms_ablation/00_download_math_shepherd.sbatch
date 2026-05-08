#!/bin/bash
#SBATCH --job-name=ms-download
#SBATCH --account=aip-azouaq
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# Download Math-Shepherd to the HF cache, then export a local JSONL so future
# jobs can run fully offline.

set -euo pipefail

STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"
OUT_DIR="$SCRATCH_BASE/ms_ablation"

mkdir -p "$OUT_DIR" results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
# Intentionally NOT setting HF_DATASETS_OFFLINE so the download goes through.

echo "Downloading peiyi9979/Math-Shepherd …"
python3 - <<'PYEOF'
import os, json
from datasets import load_dataset
from pathlib import Path

cache_dir = f"{os.environ['STORE']}/hf_cache"
out_file  = f"{os.environ['SCRATCH']}/cot-checker/ms_ablation/math_shepherd_gsm8k.jsonl"

ds = load_dataset("peiyi9979/Math-Shepherd", split="train", cache_dir=cache_dir)
print(f"Total entries: {len(ds)}")

gsm8k = [row for row in ds if row.get("task") == "GSM8K"]
print(f"GSM8K entries: {len(gsm8k)}")

Path(out_file).parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    for row in gsm8k:
        f.write(json.dumps(row) + "\n")

print(f"Saved {len(gsm8k)} GSM8K entries -> {out_file}")
PYEOF

echo "Done. Re-run the pipeline with:"
echo "  bash scripts/tamia/jobs/ms_ablation/submit_pipeline.sh"
