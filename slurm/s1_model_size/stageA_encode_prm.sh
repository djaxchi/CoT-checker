#!/bin/bash
#SBATCH --job-name=s1ms_A_encprm
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --array=0-3
#SBATCH --time=08:00:00
#SBATCH --output=%x-%A_%a.out
#
# Stage A: encode the frozen PRM800K 40K-train + 1K-val split into dense
# Qwen2.5 hidden states, sharded 4 ways across H100s (one shard per array task).
# Each task first runs the no-truncation token-length audit, then encodes its
# deterministic shard (global_index %% 4 == task) of BOTH splits.
#
# Required env (exported by submit_model_dag.sh): TAG
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/models.env"
# shellcheck disable=SC1091
source "$HERE/_common.sh"

: "${TAG:?TAG must be exported by the launcher}"
MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
BS="${S1MS_BATCH[$TAG]}"
MODEL_DIR="$RUNS_ROOT/$TAG"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
export S1MS_MODEL_ID_CUR="$MODEL_ID" S1MS_STAGE="A_encode_prm" S1MS_SHARD="$SHARD"

SHARD_DIR="$MODEL_DIR/prm800k_encode_shards/shard_0${SHARD}"
LOG_DIR="$MODEL_DIR/logs"
mkdir -p "$SHARD_DIR" "$LOG_DIR"
LOG="$LOG_DIR/stageA_encode_prm_shard${SHARD}.log"

s1ms_env_setup
echo "[A] TAG=$TAG model=$MODEL_ID shard=$SHARD/4 start_bs=$BS" | tee "$LOG"
s1ms_ensure_model_cached "$MODEL_ID" | tee -a "$LOG"
s1ms_venv

# Verify the frozen S1 splits exist before doing anything else.
for f in "$PRM_TRAIN_JSONL" "$PRM_VAL_JSONL"; do
  if [[ ! -f "$PRM_SPLIT_DIR/$f" ]]; then
    echo "[A] FATAL: frozen PRM800K split missing: $PRM_SPLIT_DIR/$f" | tee -a "$LOG" >&2
    echo "[A] Set PRM_SPLIT_DIR to the directory that produced the S1 DenseLinear row." | tee -a "$LOG" >&2
    exit 1
  fi
done

# ---- Pre-encode no-truncation audit (fails loudly if anything exceeds ctx).
PB_SPECS=()
for s in "${S1MS_SUBSETS[@]}"; do PB_SPECS+=("$s:$(s1ms_resolve_pb_raw "$s")"); done
AUDIT_JSON="$LOG_DIR/length_audit_shard${SHARD}.json"
python scripts/s1ms_audit_token_lengths.py \
  --model_name_or_path "$MODEL_ID" --local_files_only \
  --prm_split_dir "$PRM_SPLIT_DIR" \
  --prm_splits "$PRM_TRAIN_JSONL" "$PRM_VAL_JSONL" \
  --pb_subset "${PB_SPECS[@]}" \
  --out_json "$AUDIT_JSON" 2>&1 | tee -a "$LOG"
if [[ "$SHARD" == "0" ]]; then cp "$AUDIT_JSON" "$MODEL_DIR/length_audit.json"; fi

# ---- Encode this shard of both PRM800K splits (no truncation; OOM auto-retry).
run_with_oom_retry "$BS" "$LOG" \
  python scripts/encode_prm800k_hidden_states.py \
    --data_dir "$PRM_SPLIT_DIR" \
    --out_dir "$SHARD_DIR" \
    --model_name_or_path "$MODEL_ID" --local_files_only \
    --run_name "s1ms_${TAG}_prm_shard${SHARD}" \
    --max_seq_len -1 \
    --shard_idx "$SHARD" --num_shards 4 \
    --batch_size __BS__ \
    --model_dtype float16 --save_dtype float16 \
    --splits "${PRM_TRAIN_JSONL}:probe_train_40k" "${PRM_VAL_JSONL}:val_1k" \
    --force

echo "[A] shard $SHARD done -> $SHARD_DIR" | tee -a "$LOG"
