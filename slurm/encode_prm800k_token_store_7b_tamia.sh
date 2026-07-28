#!/bin/bash
#SBATCH --job-name=prm_tokstore_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

# Store ALL last-layer token states of every PRM800K step into the repstore
# (kind=token_seq, ~1 TB for the full train split). Whole-node H100:4, 4 in-node
# shards via CUDA_VISIBLE_DEVICES; read back with repstore.ShardedRepSplit (no
# merge). Set LIMIT_PER_FILE=<N> for a smoke run. Lives in SCRATCH (regenerable).

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
DATA_DIR="$RUN_ROOT/data"
REP_ROOT="$RUN_ROOT/repstore/tokens_last_layer"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"

TRAIN_STEM="${TRAIN_STEM:-probe_train_full}"
VAL_STEM="${VAL_STEM:-val_5k}"
TEST_STEM="${TEST_STEM:-test_2k}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SHARDS="${NUM_SHARDS:-4}"
LIMIT_PER_FILE="${LIMIT_PER_FILE:-0}"

mkdir -p "$REP_ROOT" "$LOG_DIR"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
LOG_FILE="$LOG_DIR/tokstore_7b-${SLURM_JOB_ID:-$$}.log"

# For a smoke run, only the small splits (val/test) to keep it fast.
if [[ "$LIMIT_PER_FILE" != "0" ]]; then
  SPLITS=("prm800k_${TEST_STEM}.jsonl:${TEST_STEM}")
else
  SPLITS=(
    "prm800k_${TRAIN_STEM}.jsonl:${TRAIN_STEM}"
    "prm800k_${VAL_STEM}.jsonl:${VAL_STEM}"
    "prm800k_${TEST_STEM}.jsonl:${TEST_STEM}"
  )
fi

cat <<BANNER
================================================================
job          : ${SLURM_JOB_NAME:-prm_tokstore_7b}  job_id: ${SLURM_JOB_ID:-N/A}
git_commit   : $GIT_COMMIT
rep_root     : $REP_ROOT
model        : $MODEL_NAME_OR_PATH  (HF_HOME=$HF_CACHE)
splits       : ${SPLITS[*]}
shards       : $NUM_SHARDS  batch: $BATCH_SIZE  limit_per_file: $LIMIT_PER_FILE
log_file     : $LOG_FILE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

LIMIT_ARG=()
if [[ "$LIMIT_PER_FILE" != "0" ]]; then LIMIT_ARG=(--limit_per_file "$LIMIT_PER_FILE"); fi

pids=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  echo "[launch] shard $i -> GPU $i" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=$i python scripts/encode_prm800k_token_store.py \
    --data_dir "$DATA_DIR" \
    --rep_root "$REP_ROOT" \
    --splits "${SPLITS[@]}" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --local_files_only \
    --layer -1 --max_seq_len 2048 \
    --batch_size "$BATCH_SIZE" \
    --model_dtype float16 \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" \
    "${LIMIT_ARG[@]}" >>"$LOG_FILE" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ "$fail" == "1" ]]; then echo "[FATAL] a shard failed; see $LOG_FILE" | tee -a "$LOG_FILE"; exit 1; fi

echo "[verify] store layout:" | tee -a "$LOG_FILE"
du -sh "$REP_ROOT"/* 2>/dev/null | tee -a "$LOG_FILE"
echo "[$(date)] tokstore_7b done"
