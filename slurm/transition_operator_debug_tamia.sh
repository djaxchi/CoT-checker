#!/bin/bash
#SBATCH --job-name=to_debug
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out

# Debug the Stage 2 A-arm NaN: exact decode path on CUDA, bf16 vs fp16.
set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn pandas pyarrow
CUDA_VISIBLE_DEVICES=0 python scripts/transition_operator/to_train.py \
  --run_dir runs/transition_operator --arm A --seed 0 --epochs 1 \
  --local_files_only --device cuda
echo "[debug done]"
