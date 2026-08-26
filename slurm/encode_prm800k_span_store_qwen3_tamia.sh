#!/bin/bash
#SBATCH --job-name=prm_spanstore_q3
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

# Encode PRM800K step spans with Qwen3-8B-Base, straight into the compact store.
#
# Backbone change from Qwen2.5-7B: 36 layers instead of 28, hidden 4096 instead
# of 3584, and official Qwen-Scope SAEs exist for the BASE model at every layer,
# which is what the SAE arm needs and what Qwen2.5 never had.
#
# --span_only writes the pre-step boundary row plus the step's own tokens with no
# full-sequence intermediate. The old two-stage path would need ~1.1 TB at 4096
# dims, which does not fit; this lands at ~157 GiB. Byte-identical to encoding in
# full then compacting (tests/harness/test_span_only_encode.py).
#
# Whole-node H100:4 is a TamIA constraint; the four GPUs each take one shard of
# the split via CUDA_VISIBLE_DEVICES, read back with ShardedRepSplit.
#
# NO INTERNET ON COMPUTE NODES. Weights must already be in $HF_CACHE before
# submitting; the job runs fully offline and asserts the snapshot is present.
# Download on the login node first:
#   HF_HOME=/project/aip-azouaq/$USER/hf_cache hf download Qwen/Qwen3-8B-Base

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
DATA_DIR="${DATA_DIR:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
REP_ROOT="${REP_ROOT:-$RUN_ROOT/repstore/step_spans}"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"

TRAIN_STEM="${TRAIN_STEM:-probe_train_full}"
VAL_STEM="${VAL_STEM:-val_5k}"
TEST_STEM="${TEST_STEM:-test_2k}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
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
LOG_FILE="$LOG_DIR/spanstore_q3-${SLURM_JOB_ID:-$$}.log"

# Fail at once, with a message, rather than 4 shards each hanging on a download
# they cannot make.
SNAP="$HF_CACHE/hub/models--$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|--|g')/snapshots"
if [[ ! -d "$SNAP" ]] || [[ -z "$(ls -A "$SNAP" 2>/dev/null)" ]]; then
  echo "[FATAL] no local snapshot for $MODEL_NAME_OR_PATH under $SNAP" >&2
  echo "        compute nodes have no internet; download on the login node:" >&2
  echo "        HF_HOME=$HF_CACHE hf download $MODEL_NAME_OR_PATH" >&2
  exit 2
fi

if [[ "$LIMIT_PER_FILE" != "0" ]]; then
  SPLITS=("prm800k_${TEST_STEM}.jsonl:${TEST_STEM}")
else
  SPLITS=(
    "prm800k_${VAL_STEM}.jsonl:${VAL_STEM}"
    "prm800k_${TEST_STEM}.jsonl:${TEST_STEM}"
    "prm800k_${TRAIN_STEM}.jsonl:${TRAIN_STEM}"
  )
fi

cat <<BANNER
================================================================
job          : ${SLURM_JOB_NAME:-prm_spanstore_q3}  job_id: ${SLURM_JOB_ID:-N/A}
git_commit   : $GIT_COMMIT
model        : $MODEL_NAME_OR_PATH   (offline, HF_HOME=$HF_CACHE)
rep_root     : $REP_ROOT   (span-only, no full-sequence intermediate)
data_dir     : $DATA_DIR   (frozen splits, reused from the previous backbone)
splits       : ${SPLITS[*]}
shards       : $NUM_SHARDS on ${SLURM_GPUS_ON_NODE:-4} GPUs   batch: $BATCH_SIZE
log_file     : $LOG_FILE
================================================================
BANNER
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
df -h "$(dirname "$REP_ROOT")" | tail -1

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

LIMIT_ARG=()
if [[ "$LIMIT_PER_FILE" != "0" ]]; then LIMIT_ARG=(--limit_per_file "$LIMIT_PER_FILE"); fi

pids=(); tags=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  echo "[launch] shard $i -> GPU $i" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=$i python scripts/encode_prm800k_token_store.py \
    --data_dir "$DATA_DIR" \
    --rep_root "$REP_ROOT" \
    --splits "${SPLITS[@]}" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --local_files_only \
    --span_only \
    --layer -1 --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --model_dtype "${MODEL_DTYPE:-bfloat16}" \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" \
    "${LIMIT_ARG[@]}" >>"$LOG_FILE" 2>&1 &
  pids+=($!); tags+=("shard_$i")
done

fail=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[ok] ${tags[$j]}"; else
    echo "[FAIL] ${tags[$j]}"; fail=1
  fi
done
if [[ "$fail" == "1" ]]; then
  echo "[FATAL] a shard failed; tail of $LOG_FILE:" >&2
  tail -30 "$LOG_FILE" >&2
  exit 1
fi

echo "[verify] store layout:" | tee -a "$LOG_FILE"
du -sh "$REP_ROOT"/* 2>/dev/null | tee -a "$LOG_FILE"
echo "[$(date)] spanstore_q3 done"
