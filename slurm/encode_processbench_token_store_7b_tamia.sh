#!/bin/bash
#SBATCH --job-name=pb_tokstore_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# ProcessBench token store (all 4 subsets) at Qwen2.5-7B base into the repstore.
# Serves both dense_last and delta by offline derivation. 4-GPU in-node shards.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench}"
REP_ROOT="$RUN_ROOT/repstore/pb_tokens_last_layer"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SUBSETS="${SUBSETS:-gsm8k math olympiadbench omnimath}"

mkdir -p "$REP_ROOT" "$LOG_DIR"
export HF_HOME="$HF_CACHE"; export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
LOG_FILE="$LOG_DIR/pb_tokstore_7b-${SLURM_JOB_ID:-$$}.log"

# Build subset:rawfile specs from whatever jsonl exist.
SPECS=()
for s in $SUBSETS; do
  f="$PB_DIR/processbench_${s}.jsonl"
  [[ -f "$f" ]] && SPECS+=("${s}:${f}") || echo "[warn] missing $f" | tee -a "$LOG_FILE"
done
[[ ${#SPECS[@]} -gt 0 ]] || { echo "[FATAL] no PB subsets"; exit 2; }
echo "[plan] $(date -Iseconds) specs: ${SPECS[*]}" | tee -a "$LOG_FILE"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

pids=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/encode_processbench_token_store.py \
    --raw_specs "${SPECS[@]}" \
    --rep_root "$REP_ROOT" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --layer -1 --max_seq_len 2048 --batch_size "$BATCH_SIZE" \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" >>"$LOG_FILE" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[FATAL] a shard failed"; exit 1; }

echo "[verify]" | tee -a "$LOG_FILE"; du -sh "$REP_ROOT"/* 2>/dev/null | tee -a "$LOG_FILE"
echo "[$(date)] pb_tokstore_7b done"
