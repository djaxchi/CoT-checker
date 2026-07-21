#!/bin/bash
#SBATCH --job-name=das_fit
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# DAS branch-subspace, Phase 3: fit the shared-U subspace. Extracts L12 + L20 span
# caches (parallel), then trains U for the L12 sweep k in {8,16,32,64} x seeds
# {0,1,2} across the 4 GPUs, and reports held-out recovery vs a same-k random
# subspace + cross-seed subspace overlap. L20 fit is a follow-up job (FIT_LAYER=20)
# reusing cache_L20.pt.
#
# Usage:  sbatch slurm/das_branch_phase3_fit_tamia.sh
#         FIT_LAYER=20 sbatch ...   # after L12, reuse cached states

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/causal_graph}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/runs/das_train}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
FIT_LAYER="${FIT_LAYER:-12}"
KTOK="${KTOK:-8}"
EPOCHS="${EPOCHS:-12}"

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

echo "=== job ${SLURM_JOB_ID:-manual} host $(hostname) fit_layer $FIT_LAYER $(date -Iseconds) ==="

FIT () { python scripts/das_branch/das_fit.py --run_dir "$RUN_DIR" --out_dir "$OUT_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only "$@"; }

# ---- extract caches for both candidate layers (once) -------------------------
if [ ! -f "$OUT_DIR/cache_L12.pt" ]; then
  CUDA_VISIBLE_DEVICES=0 FIT --mode extract --layer 12 --k_tokens "$KTOK" \
    > "das_fit_extract_L12-${SLURM_JOB_ID:-manual}.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 FIT --mode extract --layer 20 --k_tokens "$KTOK" \
    > "das_fit_extract_L20-${SLURM_JOB_ID:-manual}.log" 2>&1 &
  wait
fi

# ---- fit sweep: k x seed across 4 GPUs, waves of 4 ---------------------------
configs=(); for k in 8 16 32 64; do for s in 0 1 2; do configs+=("$k $s"); done; done
i=0; pids=()
for cfg in "${configs[@]}"; do
  set -- $cfg; k=$1; s=$2; g=$((i % 4))
  CUDA_VISIBLE_DEVICES=$g FIT --mode fit --layer "$FIT_LAYER" --k_sub "$k" \
    --seed "$s" --epochs "$EPOCHS" \
    > "das_fit_L${FIT_LAYER}_k${k}_s${s}-${SLURM_JOB_ID:-manual}.log" 2>&1 &
  pids+=($!); i=$((i + 1))
  (( i % 4 == 0 )) && { for p in "${pids[@]}"; do wait "$p"; done; pids=(); }
done
for p in "${pids[@]}"; do wait "$p"; done

python scripts/das_branch/das_fit.py --mode report --out_dir "$OUT_DIR"
echo "[das_fit] done $(date -Iseconds); gates -> $OUT_DIR/gates_das.json"
