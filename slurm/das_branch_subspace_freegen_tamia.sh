#!/bin/bash
#SBATCH --job-name=das_fgsub
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# Behavioural test of the learned DAS subspace: does the k=8 L12 subspace move the
# FREE-GENERATION solve rate (not just the teacher-forced margin)? Conditions per
# held-out fork: wrong / correct / oracle(full span) / das(U interchange) /
# random(same-k). Reuses cache_L12.pt + U_L12_k8_s0.pt from the fit sweep.
#
# Usage:  sbatch slurm/das_branch_subspace_freegen_tamia.sh
#         K_SUB=16 SEED=1 sbatch ...

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/causal_graph}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/runs/das_train}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYER="${LAYER:-12}"; K_SUB="${K_SUB:-8}"; SEED="${SEED:-0}"
MAX_EVAL="${MAX_EVAL:-160}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
test -e "$OUT_DIR/U_L${LAYER}_k${K_SUB}_s${SEED}.pt" || { echo "missing trained U"; exit 1; }

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn pandas pyarrow

echo "=== fgsub L$LAYER k$K_SUB seed$SEED $(hostname) $(date -Iseconds) ==="
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python scripts/das_branch/das_subspace_freegen.py \
    --run_dir "$RUN_DIR" --out_dir "$OUT_DIR" --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --layer "$LAYER" --k_sub "$K_SUB" --seed "$SEED" --max_eval "$MAX_EVAL" \
    --local_files_only --shard_id $g --num_shards 4 \
    > "das_fgsub_shard$g-${SLURM_JOB_ID:-manual}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

python scripts/das_branch/das_subspace_freegen.py --out_dir "$OUT_DIR" \
  --layer "$LAYER" --k_sub "$K_SUB" --merge
echo "[das_fgsub] done $(date -Iseconds); gates -> $OUT_DIR/gates_fgsub_L${LAYER}_k${K_SUB}.json"
