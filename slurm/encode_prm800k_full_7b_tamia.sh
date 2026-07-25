#!/bin/bash
#SBATCH --job-name=prm800k_encode_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

# Encode PRM800K {train_full, val, test} last-token hidden states at Qwen2.5-7B
# base into the dense-full cache contract {stem}_h.npy / {stem}_y.npy.
# Whole-node H100:4, sharded IN-NODE via CUDA_VISIBLE_DEVICES (4 shards), then
# merged deterministically. Set LIMIT_PER_FILE=<N> for a smoke run.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
DATA_DIR="$RUN_ROOT/data"
CACHE_DIR="$RUN_ROOT/cache/qwen2_5_7b"
SHARD_ROOT="$CACHE_DIR/shards"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"

TRAIN_STEM="${TRAIN_STEM:-probe_train_full}"
VAL_STEM="${VAL_STEM:-val_5k}"
TEST_STEM="${TEST_STEM:-test_2k}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_SHARDS="${NUM_SHARDS:-4}"
LIMIT_PER_FILE="${LIMIT_PER_FILE:-0}"   # >0 => smoke

mkdir -p "$CACHE_DIR" "$SHARD_ROOT" "$LOG_DIR"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
LOG_FILE="$LOG_DIR/encode_full_7b-${SLURM_JOB_ID:-$$}.log"

SPLITS=(
  "prm800k_${TRAIN_STEM}.jsonl:${TRAIN_STEM}"
  "prm800k_${VAL_STEM}.jsonl:${VAL_STEM}"
  "prm800k_${TEST_STEM}.jsonl:${TEST_STEM}"
)

cat <<BANNER
================================================================
job          : ${SLURM_JOB_NAME:-prm800k_encode_7b}
job_id       : ${SLURM_JOB_ID:-N/A}
git_commit   : $GIT_COMMIT
cache_dir    : $CACHE_DIR
model        : $MODEL_NAME_OR_PATH  (HF_HOME=$HF_CACHE)
splits       : ${SPLITS[*]}
shards       : $NUM_SHARDS   batch: $BATCH_SIZE   limit_per_file: $LIMIT_PER_FILE
log_file     : $LOG_FILE
monitor      : grep -nE "encode|shard|merge|done|ERROR|Traceback" $LOG_FILE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

LIMIT_ARG=()
if [[ "$LIMIT_PER_FILE" != "0" ]]; then LIMIT_ARG=(--limit_per_file "$LIMIT_PER_FILE"); fi
FORCE_ARG=()
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_ARG=(--force); fi

# ---- Launch one encoder per GPU (shard i on GPU i) ----
pids=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  sdir=$(printf "%s/shard_%02d" "$SHARD_ROOT" "$i")
  mkdir -p "$sdir"
  echo "[launch] shard $i/$NUM_SHARDS -> GPU $i -> $sdir" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=$i python scripts/encode_prm800k_hidden_states.py \
    --data_dir "$DATA_DIR" \
    --out_dir "$sdir" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --local_files_only \
    --run_name "dense_full_7b_v1_shard${i}" \
    --max_seq_len 2048 \
    --batch_size "$BATCH_SIZE" \
    --model_dtype float16 --save_dtype float16 \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" \
    --splits "${SPLITS[@]}" \
    "${LIMIT_ARG[@]}" "${FORCE_ARG[@]}" \
    >>"$LOG_FILE" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ "$fail" == "1" ]]; then
  echo "[FATAL] a shard encoder failed; see $LOG_FILE" | tee -a "$LOG_FILE"; exit 1
fi

# ---- Merge shards per stem into the dense-full contract ----
for stem in "$TRAIN_STEM" "$VAL_STEM" "$TEST_STEM"; do
  echo "[merge] $stem" | tee -a "$LOG_FILE"
  python scripts/merge_prm800k_encoded_shards.py \
    --shard_root "$SHARD_ROOT" \
    --stem "$stem" \
    --out_dir "$CACHE_DIR" \
    "${FORCE_ARG[@]}" >>"$LOG_FILE" 2>&1
done

echo "[verify] cache contents:" | tee -a "$LOG_FILE"
ls -la "$CACHE_DIR"/*.npy | tee -a "$LOG_FILE"
echo "[$(date)] encode_full_7b done"
