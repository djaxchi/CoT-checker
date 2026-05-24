#!/bin/bash
#SBATCH --job-name=dense_linear_full
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load StdEnv/2023 python/3.12

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_v1}"
CACHE_DIR="$RUN_ROOT/cache/qwen2_5_1_5b"
PB_ROOT="$RUN_ROOT/cache/qwen2_5_1_5b_processbench"
OUT_ROOT="$RUN_ROOT/runs/dense_linear"
LOG_DIR="$RUN_ROOT/logs"

TRAIN_STEM="${TRAIN_STEM:-probe_train_400k}"
VAL_STEM="${VAL_STEM:-val_10k}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.01}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-512}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
LOG_FILE="$LOG_DIR/dense_linear_full-${SLURM_JOB_ID:-$$}.log"

# Build PB target specs from whatever exists under $PB_ROOT
PB_SPECS=()
for sub_dir in "$PB_ROOT"/*; do
  [[ -d "$sub_dir" ]] || continue
  name="$(basename "$sub_dir")"
  h="$sub_dir/pb_step_h.npy"
  meta="$sub_dir/pb_step_meta.jsonl"
  if [[ -f "$h" && -f "$meta" ]]; then
    PB_SPECS+=("${name}:${h}:${meta}")
  fi
done

if [[ ${#PB_SPECS[@]} -eq 0 ]]; then
  echo "[FATAL] No PB caches under $PB_ROOT (run encode_processbench_full first)" >&2
  exit 2
fi

cat <<BANNER
================================================================
job          : ${SLURM_JOB_NAME:-dense_linear_full}
job_id       : ${SLURM_JOB_ID:-N/A}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
cache_dir    : $CACHE_DIR
out_dir      : $OUT_ROOT
train_stem   : $TRAIN_STEM
val_stem     : $VAL_STEM
pb_specs     : ${PB_SPECS[*]}
threshold    : $THRESHOLD_GRID
log_file     : $LOG_FILE
monitor      :
  grep -nE "train_metrics|F1_PB|Acc_error|Acc_correct|oracle|threshold|ERROR|Traceback" $LOG_FILE
  ls -1 $OUT_ROOT/eval_metrics_*.json
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pyyaml

CMD=(python scripts/train_easy_probe_method.py
  --method dense_linear
  --cache_dir "$CACHE_DIR"
  --out_dir "$OUT_ROOT"
  --probe_train_stem "$TRAIN_STEM"
  --val_stem "$VAL_STEM"
  --skip_size_asserts
  --pb_specs "${PB_SPECS[@]}"
  --threshold_grid "$THRESHOLD_GRID"
  --seed "$SEED"
  --epochs_probe 50
  --batch_size "$BATCH_SIZE"
  --lr_probe 1e-3)

echo "[CMD] ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

# Leaderboard
PRM_MANIFEST="$CACHE_DIR/encoding_manifest.json"
PB_MANIFEST="$PB_ROOT/combined/encoding_manifest_pb.json"
python scripts/merge_dense_full_leaderboard.py \
  --runs_dir "$RUN_ROOT/runs" \
  --out_csv  "$RUN_ROOT/runs/leaderboard_dense_full.csv" \
  --out_md   "$RUN_ROOT/runs/leaderboard_dense_full.md" \
  --prm_manifest "$PRM_MANIFEST" \
  --pb_manifest  "$PB_MANIFEST" \
  --train_stem "$TRAIN_STEM" \
  --val_stem   "$VAL_STEM" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] dense_linear_full done"
