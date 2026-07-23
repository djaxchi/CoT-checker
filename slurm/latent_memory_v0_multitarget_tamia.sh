#!/bin/bash
#SBATCH --job-name=lm_mt
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --output=%x-%j.out

# latent_memory_v0 multi-target reasoning-state oracle: can ONE latent, optimised jointly
# on the final answer + several cue-anchored intermediate values, serve them all, and how
# does that scale with width m? 4 in-node shards, then a CPU merge + capacity-curve plot.
#
# Usage:  sbatch slurm/latent_memory_v0_multitarget_tamia.sh
# Design: docs/latent_memory_v0_plan.md

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/latent_memory_v0}"
TRACES="${TRACES:-$PROJECT_ROOT/runs/causal_graph/traces_forks.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYERS="${LAYERS:-0,20}"
M_LIST="${M_LIST:-1,2,4,8,16}"
N_TARGETS="${N_TARGETS:-3}"
LIMIT="${LIMIT:-240}"
SPLIT="${SPLIT:-test}"
EPOCHS="${EPOCHS:-60}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
test -e "$TRACES" || { echo "missing traces $TRACES"; exit 1; }
mkdir -p "$RUN_DIR"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy pandas matplotlib huggingface_hub

cat <<BANNER
================================================================
job     : ${SLURM_JOB_NAME:-lm_mt}  id ${SLURM_JOB_ID:-N/A}
host    : $(hostname)   date $(date -Iseconds)
run_dir : $RUN_DIR  layers : $LAYERS  m : $M_LIST  n_targets : $N_TARGETS  limit : $LIMIT
================================================================
BANNER

pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python -m scripts.latent_memory.lm_multitarget \
    --traces "$TRACES" --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --run_dir "$RUN_DIR" --layers "$LAYERS" --m_list "$M_LIST" --n_targets "$N_TARGETS" \
    --limit "$LIMIT" --split "$SPLIT" --epochs "$EPOCHS" \
    --shard_id $g --num_shards 4 \
    > "lm_mt_shard$g-${SLURM_JOB_ID:-manual}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

python -m scripts.latent_memory.lm_multitarget_summary --run_dir "$RUN_DIR"
echo "[lm_mt] done $(date -Iseconds)"
