#!/bin/bash
#SBATCH --job-name=prm800k_heldout_encode
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Encode the fresh problem-disjoint PRM800K held-out TEST set with the SAME 7B
# readout as val_1k (last layer, last token of candidate_step), writing the
# arrays straight into the run's merged/ dir so the loader + analysis scripts
# pick them up via --stem prm800k_heldout_test.
#
# Build the jsonl first with scripts/build_prm800k_heldout_test.py.
#
# Usage:
#   sbatch slurm/encode_prm800k_heldout_test_tamia.sh
#   FORCE=1 sbatch slurm/encode_prm800k_heldout_test_tamia.sh   # overwrite

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
DATA_DIR="${DATA_DIR:-$SCRATCH/cot_mech/prestudy_v1/data}"
OUT_DIR="${OUT_DIR:-$HOME/CoT-checker/runs/s1_model_size_dense/qwen2_5_7b/merged}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
STEM="${STEM:-prm800k_heldout_test}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"

mkdir -p "$OUT_DIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

cat <<BANNER
================================================================
job          : ${SLURM_JOB_NAME:-prm800k_heldout_encode}
job_id       : ${SLURM_JOB_ID:-N/A}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
model        : $MODEL_NAME_OR_PATH
data_dir     : $DATA_DIR
out_dir      : $OUT_DIR
split        : ${STEM}.jsonl:${STEM}
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
  --out_dir "$OUT_DIR"
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --local_files_only
  --run_name "heldout_test_qwen2_5_7b_${STEM}"
  --max_seq_len -1
  --batch_size "$BATCH_SIZE"
  --model_dtype float16
  --save_dtype float16
  --splits "${STEM}.jsonl:${STEM}"
  "${EXTRA[@]}")

echo "[CMD] ${CMD[*]}"
"${CMD[@]}"

echo "[$(date)] heldout encode done -> $OUT_DIR/${STEM}_{h,y,meta}"
