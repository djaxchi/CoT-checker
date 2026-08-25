#!/bin/bash
#SBATCH --job-name=pb_spanstore_q3
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# ProcessBench step spans with Qwen3-8B-Base, all four subsets, straight into the
# compact store (--span_only). Same protocol notes as the PRM800K job: whole-node
# H100:4 with one shard per GPU, and NO INTERNET on compute nodes, so the weights
# must already be in $HF_CACHE before submitting.
#
# ProcessBench steps run longer than PRM800K ones (56 to 94 tokens per step
# against 38.8), so this store lands nearer 20% of a full-sequence store rather
# than 14%; still ~17 GiB rather than ~110 GiB.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench_full}"
REP_ROOT="${REP_ROOT:-$RUN_ROOT/repstore/pb_step_spans}"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
SUBSETS="${SUBSETS:-gsm8k math olympiadbench omnimath}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

mkdir -p "$REP_ROOT" "$LOG_DIR"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
LOG_FILE="$LOG_DIR/pb_spanstore_q3-${SLURM_JOB_ID:-$$}.log"

SNAP="$HF_CACHE/hub/models--$(echo "$MODEL_NAME_OR_PATH" | tr '/' '-')/snapshots"
if [[ ! -d "$SNAP" ]] || [[ -z "$(ls -A "$SNAP" 2>/dev/null)" ]]; then
  echo "[FATAL] no local snapshot for $MODEL_NAME_OR_PATH under $SNAP" >&2
  echo "        compute nodes have no internet; download on the login node first" >&2
  exit 2
fi

SPECS=()
for s in $SUBSETS; do
  f="$PB_DIR/processbench_${s}.jsonl"
  if [[ -f "$f" ]]; then SPECS+=("${s}:${f}"); else echo "[warn] missing $f"; fi
done
[[ ${#SPECS[@]} -gt 0 ]] || { echo "[FATAL] no ProcessBench jsonl under $PB_DIR" >&2; exit 2; }

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-pb_spanstore_q3}  job_id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH  (offline)
rep_root   : $REP_ROOT  (span-only)
specs      : ${SPECS[*]}
shards     : $NUM_SHARDS on ${SLURM_GPUS_ON_NODE:-4} GPUs   batch: $BATCH_SIZE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

pids=(); tags=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  echo "[launch] shard $i -> GPU $i" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=$i python scripts/encode_processbench_token_store.py \
    --raw_specs "${SPECS[@]}" \
    --rep_root "$REP_ROOT" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --local_files_only \
    --span_only \
    --layer -1 --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --model_dtype "${MODEL_DTYPE:-bfloat16}" \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" >>"$LOG_FILE" 2>&1 &
  pids+=($!); tags+=("shard_$i")
done

fail=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[ok] ${tags[$j]}"; else echo "[FAIL] ${tags[$j]}"; fail=1; fi
done
if [[ "$fail" == "1" ]]; then tail -30 "$LOG_FILE" >&2; exit 1; fi

du -sh "$REP_ROOT"/* 2>/dev/null | tee -a "$LOG_FILE"
echo "[$(date)] pb_spanstore_q3 done"
