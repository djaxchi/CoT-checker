#!/bin/bash
#SBATCH --job-name=prm800k_build_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Build the frozen PRM800K splits for the unified 7B harness:
#   - prm800k_probe_train_full.jsonl  (all balanced +/-1, the training pool)
#   - prm800k_val_5k.jsonl            (balanced val, threshold selection)
#   - prm800k_test_2k.jsonl           (small balanced in-domain test)
# Splits are problem-id disjoint across train/val/test. Tokenizer-only (CPU);
# the same JSONL is later encoded at 7B. Qwen2.5 share a tokenizer, so the
# length filter is identical to the 1.5B build.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
RAW_PRM800K_DIR="${RAW_PRM800K_DIR:-$SCRATCH/cot_mech/raw/prm800k}"
DATA_DIR="$RUN_ROOT/data"
LOG_DIR="$RUN_ROOT/logs"
# 7B base weights are cached in $STORE/hf_cache, not $SCRATCH/hf_cache.
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"

VAL_POS="${VAL_POS:-2500}"
VAL_NEG="${VAL_NEG:-2500}"
TEST_POS="${TEST_POS:-1000}"
TEST_NEG="${TEST_NEG:-1000}"

mkdir -p "$DATA_DIR" "$LOG_DIR"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
LOG_FILE="$LOG_DIR/build_full_7b-${SLURM_JOB_ID:-$$}.log"

cat <<BANNER
================================================================
job          : ${SLURM_JOB_NAME:-prm800k_build_7b}
job_id       : ${SLURM_JOB_ID:-N/A}
git_commit   : $GIT_COMMIT
out_dir      : $DATA_DIR
model        : $MODEL_NAME_OR_PATH  (HF_HOME=$HF_CACHE)
splits       : train_full + val_$(( (VAL_POS+VAL_NEG)/1000 ))k + test_$(( (TEST_POS+TEST_NEG)/1000 ))k
log_file     : $LOG_FILE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index transformers numpy

EXTRA=()
if [[ "${FORCE:-0}" == "1" ]]; then EXTRA+=(--force); fi

CMD=(python scripts/build_prm800k_full.py
  --raw_dir "$RAW_PRM800K_DIR"
  --out_dir "$DATA_DIR"
  --tokenizer_name_or_path "$MODEL_NAME_OR_PATH"
  --local_files_only
  --run_name "dense_full_7b_v1_qwen2_5_7b"
  --seed 42
  --max_seq_len 2048
  --train_sizes 400000
  --full
  --val_pos "$VAL_POS" --val_neg "$VAL_NEG"
  --val_name "val_$(( (VAL_POS+VAL_NEG)/1000 ))k"
  --test_pos "$TEST_POS" --test_neg "$TEST_NEG"
  --test_name "test_$(( (TEST_POS+TEST_NEG)/1000 ))k"
  "${EXTRA[@]}")

echo "[CMD] ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "[$(date)] build_full_7b done"
