#!/bin/bash
#SBATCH --job-name=pb_build_subsets
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out

# Materialize the missing ProcessBench subsets (olympiadbench, omnimath) as
# raw-trace JSONL from the cached Qwen/ProcessBench dataset (offline).

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench}"
SUBSETS="${SUBSETS:-olympiadbench omnimath}"
export HF_HOME="/project/aip-azouaq/$USER/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "$PROJECT_ROOT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index datasets numpy pyarrow

python scripts/build_processbench_subsets_jsonl.py --subsets $SUBSETS --out_dir "$PB_DIR"
echo "[build_pb_subsets] done"; ls -la "$PB_DIR"/*.jsonl
