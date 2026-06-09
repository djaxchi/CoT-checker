#!/bin/bash
#SBATCH --job-name=s1ms_A_encprm
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#
# Stage A: encode the frozen PRM800K 40K-train + 1K-val split into dense
# Qwen2.5 hidden states. TamIA allocates h100 GPUs by whole node, so this job
# grabs all 4 GPUs and fans out 4 background workers (one per GPU), each
# encoding its deterministic shard (global_index %% 4 == g) of BOTH splits.
# A pre-encode no-truncation audit runs once up front.
#
# Required env (exported by submit_model_dag.sh): TAG
set -euo pipefail

# sbatch copies this script to a spool dir, so $BASH_SOURCE does not point at
# the repo. The launcher exports S1MS_DIR with the real path; fall back to
# dirname only when run directly (not via sbatch).
HERE="${S1MS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "$HERE/models.env"
# shellcheck disable=SC1091
source "$HERE/_common.sh"

: "${TAG:?TAG must be exported by the launcher}"
MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
BS="${S1MS_BATCH[$TAG]}"
MODEL_DIR="$RUNS_ROOT/$TAG"
export S1MS_MODEL_ID_CUR="$MODEL_ID" S1MS_STAGE="A_encode_prm"

LOG_DIR="$MODEL_DIR/logs"
mkdir -p "$LOG_DIR" "$MODEL_DIR/prm800k_encode_shards"
LOG="$LOG_DIR/stageA_encode_prm.log"

s1ms_env_setup
echo "[A] TAG=$TAG model=$MODEL_ID 4-GPU fan-out start_bs=$BS" | tee "$LOG"
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
python scripts/s1ms_audit_token_lengths.py \
  --model_name_or_path "$MODEL_ID" --local_files_only \
  --prm_split_dir "$PRM_SPLIT_DIR" \
  --prm_splits "$PRM_TRAIN_JSONL" "$PRM_VAL_JSONL" \
  --pb_subset "${PB_SPECS[@]}" \
  --out_json "$MODEL_DIR/length_audit.json" 2>&1 | tee -a "$LOG"

# ---- Fan out one encode worker per GPU (shared helper in _common.sh).
s1ms_encode_prm_fanout || { echo "[A] FATAL: a PRM shard failed (model=$MODEL_ID)" | tee -a "$LOG" >&2; exit 1; }
echo "[A] all 4 PRM shards done -> $MODEL_DIR/prm800k_encode_shards" | tee -a "$LOG"
