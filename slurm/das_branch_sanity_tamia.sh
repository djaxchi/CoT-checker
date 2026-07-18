#!/bin/bash
#SBATCH --job-name=das_sanity
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# DAS branch-subspace live-patch check. Before trusting the null oracle (Phase 1b),
# confirm the boundary patch is materially active through the forward path: patching
# the wrong branch with the correct branch's boundary state should move the
# NEXT-TOKEN distribution toward the correct branch (S6 Stage-0 measured 0.90-1.0
# next-token recovery). High recovery here => the patch is live and the null
# answer-level oracle is a real scientific result, not an integration bug.
#
# Usage:  sbatch slurm/das_branch_sanity_tamia.sh

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/causal_graph}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/runs/das_branch}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYERS="${LAYERS:-12,20,26}"
SANITY_N="${SANITY_N:-64}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn pandas pyarrow

CUDA_VISIBLE_DEVICES=0 python scripts/das_branch/das_oracle.py \
  --mode sanity --run_dir "$RUN_DIR" --out_dir "$OUT_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --layers "$LAYERS" \
  --sanity_n "$SANITY_N" --local_files_only

echo "[das_sanity] done $(date -Iseconds); -> $OUT_DIR/sanity_next_token.json"
