#!/bin/bash
#SBATCH --job-name=harness_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Unified 7B harness run: train one representation on PRM800K, select threshold
# on val, report in-domain PRM800K test (AUROC + F1) then OOD ProcessBench F1.
# Only the representation varies between runs (METHOD). Whole-node alloc is a
# TamIA constraint; the run itself uses a single GPU.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
# REP_TAG selects the representation cache; empty = the original dense_last cache.
REP_TAG="${REP_TAG:-}"
CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/cache/qwen2_5_7b${REP_TAG:+_$REP_TAG}}"
PB_ROOT="${PB_ROOT:-$RUN_ROOT/cache/qwen2_5_7b_processbench}"
METHOD="${METHOD:-dense_linear}"
RUN_TAG="${RUN_TAG:-${REP_TAG:-dense_last}}"
OUT_ROOT="$RUN_ROOT/runs/${RUN_TAG}_${METHOD}"
LOG_DIR="$RUN_ROOT/logs"

TRAIN_STEM="${TRAIN_STEM:-probe_train_full}"
VAL_STEM="${VAL_STEM:-val_5k}"
TEST_STEM="${TEST_STEM:-test_2k}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.01}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-512}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
LOG_FILE="$LOG_DIR/harness_7b_${METHOD}-${SLURM_JOB_ID:-$$}.log"

# Build PB target specs from whatever exists under $PB_ROOT.
PB_SPECS=()
for sub_dir in "$PB_ROOT"/*; do
  [[ -d "$sub_dir" ]] || continue
  name="$(basename "$sub_dir")"
  h="$sub_dir/pb_step_h.npy"; meta="$sub_dir/pb_step_meta.jsonl"
  [[ -f "$h" && -f "$meta" ]] && PB_SPECS+=("${name}:${h}:${meta}")
done
if [[ ${#PB_SPECS[@]} -eq 0 ]]; then
  echo "[FATAL] No PB caches under $PB_ROOT" >&2; exit 2
fi

cat <<BANNER
================================================================
job          : ${SLURM_JOB_NAME:-harness_7b}  job_id: ${SLURM_JOB_ID:-N/A}
git_commit   : $GIT_COMMIT
method       : $METHOD
cache_dir    : $CACHE_DIR
train/val/test: $TRAIN_STEM / $VAL_STEM / $TEST_STEM
pb_specs     : ${PB_SPECS[*]}
log_file     : $LOG_FILE
monitor      : grep -nE "in_domain|F1_PB|val_selected|oracle|AUROC|ERROR|Traceback" $LOG_FILE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pyyaml

CMD=(python scripts/train_easy_probe_method.py
  --method "$METHOD"
  --cache_dir "$CACHE_DIR"
  --out_dir "$OUT_ROOT"
  --probe_train_stem "$TRAIN_STEM"
  --val_stem "$VAL_STEM"
  --test_stem "$TEST_STEM"
  --skip_size_asserts
  --pb_specs "${PB_SPECS[@]}"
  --threshold_grid "$THRESHOLD_GRID"
  --seed "$SEED"
  --epochs_probe 50
  --batch_size "$BATCH_SIZE"
  --lr_probe 1e-3)

echo "[CMD] ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "[$(date)] harness_7b ($METHOD) done"
