#!/bin/bash
#SBATCH --job-name=lm_ab
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# latent_memory_v0 tests A + B: falsify the answer-shortcut suspected from the capacity
# oracle. A (lm_probe, GPU0): does the answer-optimised latent recall an intermediate
# value? B (lm_swap, GPU1): does injecting a donor trace's latent pull the recipient's
# answer onto the donor's? Run in parallel on two GPUs.
#
# Usage:  sbatch slurm/latent_memory_v0_ab_tamia.sh
# Design: docs/latent_memory_v0_plan.md

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/latent_memory_v0}"
TRACES="${TRACES:-$PROJECT_ROOT/runs/causal_graph/traces_forks.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYERS="${LAYERS:-0,20}"
LIMIT="${LIMIT:-300}"
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
pip install --no-index torch transformers numpy scipy pandas huggingface_hub

cat <<BANNER
================================================================
job     : ${SLURM_JOB_NAME:-lm_ab}  id ${SLURM_JOB_ID:-N/A}
host    : $(hostname)   date $(date -Iseconds)
run_dir : $RUN_DIR   layers : $LAYERS   limit : $LIMIT   split : $SPLIT
================================================================
BANNER

CUDA_VISIBLE_DEVICES=0 python -m scripts.latent_memory.lm_probe \
  --traces "$TRACES" --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --run_dir "$RUN_DIR" --layers "$LAYERS" --limit "$LIMIT" --split "$SPLIT" \
  --epochs "$EPOCHS" > "lm_probe-${SLURM_JOB_ID:-manual}.log" 2>&1 &
P_PROBE=$!

CUDA_VISIBLE_DEVICES=1 python -m scripts.latent_memory.lm_swap \
  --traces "$TRACES" --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --run_dir "$RUN_DIR" --layers "$LAYERS" --limit "$LIMIT" --split "$SPLIT" \
  --epochs "$EPOCHS" > "lm_swap-${SLURM_JOB_ID:-manual}.log" 2>&1 &
P_SWAP=$!

wait "$P_PROBE"; echo "[probe] exit $?"
wait "$P_SWAP"; echo "[swap] exit $?"

echo "=== probe summary ==="; cat "$RUN_DIR"/probe_summary_L*.json 2>/dev/null
echo "=== swap summary ===";  cat "$RUN_DIR"/swap_summary_L*.json 2>/dev/null
echo "[lm_ab] done $(date -Iseconds)"
