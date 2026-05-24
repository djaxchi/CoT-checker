#!/bin/bash
#SBATCH --job-name=pb_encode_full
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load StdEnv/2023 python/3.12

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_v1}"
PB_DIR_DEFAULT="/scratch/d/dchikhi/cot-checker/processbench"
PB_DIR="${PB_DIR:-$PB_DIR_DEFAULT}"
OUT_ROOT="$RUN_ROOT/cache/qwen2_5_1_5b_processbench"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"
BATCH_SIZE="${BATCH_SIZE:-16}"   # single-GPU (matches proven prestudy)

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
LOG_FILE="$LOG_DIR/pb_encode_full-${SLURM_JOB_ID:-$$}.log"

# Discover subsets. Defaults: gsm8k + math (jsonl preferred, json fallback).
SUBSETS=()
for sub in gsm8k math olympiadbench omnimath; do
  for ext in jsonl json; do
    f="$PB_DIR/processbench_${sub}.${ext}"
    if [[ -f "$f" ]]; then
      SUBSETS+=("${sub}:${f}")
      break
    fi
  done
done

if [[ ${#SUBSETS[@]} -eq 0 ]]; then
  echo "[FATAL] No ProcessBench files found under $PB_DIR" >&2
  exit 2
fi

cat <<BANNER
================================================================
job          : ${SLURM_JOB_NAME:-pb_encode_full}
job_id       : ${SLURM_JOB_ID:-N/A}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
out_root     : $OUT_ROOT
subsets      : ${SUBSETS[*]}
log_file     : $LOG_FILE
monitor      :
  grep -nE "multi_pb|encode_pb|Done|ERROR|Traceback" $LOG_FILE
  ls -1 $OUT_ROOT/*/pb_step_h.npy
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

EXTRA=()
if [[ "${FORCE:-0}" == "1" ]]; then EXTRA+=(--force); fi

CMD=(python scripts/encode_processbench_multi.py
  --subsets "${SUBSETS[@]}"
  --out_root "$OUT_ROOT"
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --local_files_only
  --run_name "dense_full_v1_qwen2_5_1_5b_processbench"
  --max_seq_len 2048
  --batch_size "$BATCH_SIZE"
  --model_dtype float16
  --save_dtype float16
  "${EXTRA[@]}")

echo "[CMD] ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] pb_encode_full done"
