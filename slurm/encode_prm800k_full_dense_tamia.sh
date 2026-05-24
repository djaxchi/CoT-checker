#!/bin/bash
#SBATCH --job-name=prm800k_encode_full
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_v1}"
DATA_DIR="$RUN_ROOT/data"
CACHE_DIR="$RUN_ROOT/cache/qwen2_5_1_5b"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"

# Choose which splits to encode. Defaults to 400k train + 10k val.
TRAIN_STEM="${TRAIN_STEM:-probe_train_400k}"
VAL_STEM="${VAL_STEM:-val_10k}"
BATCH_SIZE="${BATCH_SIZE:-16}"   # single-GPU (matches proven prestudy 40k throughput)

mkdir -p "$CACHE_DIR" "$LOG_DIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
LOG_FILE="$LOG_DIR/encode_full-${SLURM_JOB_ID:-$$}.log"

cat <<BANNER
================================================================
job          : ${SLURM_JOB_NAME:-prm800k_encode_full}
job_id       : ${SLURM_JOB_ID:-N/A}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
cache_dir    : $CACHE_DIR
log_file     : $LOG_FILE
splits       : prm800k_${TRAIN_STEM}.jsonl:${TRAIN_STEM} prm800k_${VAL_STEM}.jsonl:${VAL_STEM}
monitor      :
  grep -nE "encode|done|Total|avg|ERROR|Traceback" $LOG_FILE
  tail -F $LOG_FILE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

EXTRA=()
if [[ "${FORCE:-0}" == "1" ]]; then EXTRA+=(--force); fi

CMD=(python scripts/encode_prm800k_hidden_states.py
  --data_dir "$DATA_DIR"
  --out_dir "$CACHE_DIR"
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --local_files_only
  --run_name "dense_full_v1_qwen2_5_1_5b_${TRAIN_STEM}_${VAL_STEM}"
  --max_seq_len 2048
  --batch_size "$BATCH_SIZE"
  --model_dtype float16
  --save_dtype float16
  --splits "prm800k_${TRAIN_STEM}.jsonl:${TRAIN_STEM}" "prm800k_${VAL_STEM}.jsonl:${VAL_STEM}"
  "${EXTRA[@]}")

echo "[CMD] ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] encode_full done"
