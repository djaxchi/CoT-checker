#!/bin/bash
#SBATCH --job-name=prm800k_build_full
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_v1}"
RAW_PRM800K_DIR="${RAW_PRM800K_DIR:-$SCRATCH/cot_mech/raw/prm800k}"
DATA_DIR="$RUN_ROOT/data"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"

# Configurable train sizes and val sizes
TRAIN_SIZES="${TRAIN_SIZES:-200000 400000}"
VAL_POS="${VAL_POS:-5000}"
VAL_NEG="${VAL_NEG:-5000}"
EMIT_FULL_FLAG="${EMIT_FULL:-}"

mkdir -p "$DATA_DIR" "$LOG_DIR" "$HF_CACHE"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
LOG_FILE="$LOG_DIR/build_full-${SLURM_JOB_ID:-$$}.log"

cat <<BANNER
================================================================
job          : ${SLURM_JOB_NAME:-prm800k_build_full}
job_id       : ${SLURM_JOB_ID:-N/A}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
out_dir      : $DATA_DIR
log_file     : $LOG_FILE
sizes        : $TRAIN_SIZES (val_pos=$VAL_POS val_neg=$VAL_NEG)
emit_full    : ${EMIT_FULL_FLAG:-no}
monitor      :
  grep -nE "build_full|wrote|overwrite|Error|Traceback" $LOG_FILE
  grep -n manifest $LOG_FILE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index transformers numpy

EXTRA=()
if [[ -n "$EMIT_FULL_FLAG" ]]; then EXTRA+=(--full); fi
if [[ "${FORCE:-0}" == "1" ]]; then EXTRA+=(--force); fi

CMD=(python scripts/build_prm800k_full.py
  --raw_dir "$RAW_PRM800K_DIR"
  --out_dir "$DATA_DIR"
  --tokenizer_name_or_path "$MODEL_NAME_OR_PATH"
  --local_files_only
  --run_name "dense_full_v1_qwen2_5_1_5b"
  --seed 42
  --max_seq_len 2048
  --train_sizes $TRAIN_SIZES
  --val_pos "$VAL_POS"
  --val_neg "$VAL_NEG"
  "${EXTRA[@]}")

echo "[CMD] ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] build_full done"
