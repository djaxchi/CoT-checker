#!/bin/bash
# Evaluate every trained probe on every ProcessBench subset using 4 parallel
# workers (one per GPU). Each (method, subset) pair is a discrete eval job;
# round-robin across workers; merge into final leaderboards.
#
# Pipeline:
#   1. build_full_processbench_eval_jobs.py  -> jobs + per-worker shards
#   2. 4x evaluate_existing_probes_full_processbench_worker.py (parallel)
#   3. merge_full_processbench_eval_results.py -> CSV+MD leaderboards
#
# Override knobs:
#   METHODS="..."         # space-separated method list
#   NUM_WORKERS=4         # default 4
#   FORCE=1               # allow overwriting per-job and leaderboard files
#   SKIP_MISSING=1        # silently drop method/subset pairs with missing files
#   ORACLE_STEP=0.005     # fine-grained oracle sweep step (default 0.005)
#   INCLUDE_COMBINED=1    # also evaluate the pooled 'combined' subset

#SBATCH --job-name=full_pb_eval
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load StdEnv/2023 python/3.12

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/prestudy_v1}"
RUNS_DIR="$RUN_ROOT/runs"
DENSE_PB_CACHE="$RUN_ROOT/cache/qwen2_5_1_5b_processbench_full"
OUT_DIR="$RUNS_DIR/full_processbench_eval"
LOG_DIR="$RUN_ROOT/logs"
NUM_WORKERS="${NUM_WORKERS:-4}"
ORACLE_STEP="${ORACLE_STEP:-0.005}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
JID="${SLURM_JOB_ID:-$$}"
LOG_FILE="$LOG_DIR/full_pb_eval-${JID}.log"

cat <<BANNER | tee -a "$LOG_FILE"
================================================================
job          : ${SLURM_JOB_NAME:-full_pb_eval}
job_id       : ${JID}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
runs_root    : $RUNS_DIR
dense_cache  : $DENSE_PB_CACHE
out_dir      : $OUT_DIR
num_workers  : $NUM_WORKERS  (one GPU per worker)
oracle_step  : $ORACLE_STEP
log_file     : $LOG_FILE
monitor      :
  nvidia-smi
  squeue -j $JID
  grep -nE "worker|build_jobs|merge|F1=|ERROR|Traceback" $LOG_FILE | tail -200
  ls -1 $OUT_DIR/leaderboard_full_pb_*.{csv,md}
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

METHODS_DEFAULT="dense_linear random sae_positive sae_mixed sae_contrastive ssae_positive ssae_mixed ssae_contrastive ssae_contrastive_auxlr1e-3_full"
METHODS="${METHODS:-$METHODS_DEFAULT}"
# shellcheck disable=SC2206
METHODS_ARR=($METHODS)

JOBS_DIR="$OUT_DIR/_jobs"
mkdir -p "$JOBS_DIR"
SHARD_PREFIX="$JOBS_DIR/eval_jobs"

EXTRA_BUILD=()
if [[ "${SKIP_MISSING:-0}" == "1" ]]; then EXTRA_BUILD+=(--skip_missing); fi
if [[ "${INCLUDE_COMBINED:-0}" == "1" ]]; then EXTRA_BUILD+=(--include_combined); fi

echo "[$(date -Iseconds)] building job list" | tee -a "$LOG_FILE"
python scripts/build_full_processbench_eval_jobs.py \
  --runs_root "$RUNS_DIR" \
  --dense_pb_cache_root "$DENSE_PB_CACHE" \
  --out_dir "$OUT_DIR" \
  --methods "${METHODS_ARR[@]}" \
  --out_jobs_jsonl "$JOBS_DIR/eval_jobs.jsonl" \
  --num_workers "$NUM_WORKERS" \
  --shard_prefix "$SHARD_PREFIX" \
  "${EXTRA_BUILD[@]}" 2>&1 | tee -a "$LOG_FILE"

TOTAL_JOBS=$(wc -l < "$JOBS_DIR/eval_jobs.jsonl" | awk '{print $1}')
echo "[plan] total_jobs=$TOTAL_JOBS workers=$NUM_WORKERS" | tee -a "$LOG_FILE"
if [[ "$TOTAL_JOBS" -eq 0 ]]; then
  echo "[FATAL] no eval jobs built; check artifacts under $RUNS_DIR and $DENSE_PB_CACHE" \
    | tee -a "$LOG_FILE"
  exit 2
fi

WORKER_FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then WORKER_FORCE_FLAG="--force"; fi

WORKER_PIDS=()
WORKER_LOGS=()
for ((W=0; W<NUM_WORKERS; W++)); do
  GPU=$W
  JOBS_JSON="${SHARD_PREFIX}_worker_${W}.json"
  WLOG="$LOG_DIR/full_pb_eval-${JID}-worker${W}.log"
  WORKER_LOGS+=("$WLOG")
  JOBS_FOR_W=$(python -c "import json; print(len(json.load(open('$JOBS_JSON'))))")
  echo "[worker-launch] worker=$W gpu=$GPU jobs=$JOBS_FOR_W log=$WLOG" \
    | tee -a "$LOG_FILE"
  (
    echo "[worker] worker_id=$W gpu=$GPU pid=$$ start=$(date -Iseconds)"
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/evaluate_existing_probes_full_processbench_worker.py \
      --jobs_json "$JOBS_JSON" \
      --worker_id "$W" \
      --device cuda \
      --oracle_threshold_step "$ORACLE_STEP" \
      $WORKER_FORCE_FLAG
    echo "[worker] worker_id=$W rc=$? end=$(date -Iseconds)"
  ) >"$WLOG" 2>&1 &
  WORKER_PIDS+=("$!")
done

FAIL=0
for i in "${!WORKER_PIDS[@]}"; do
  PID=${WORKER_PIDS[$i]}
  if ! wait "$PID"; then
    echo "[worker-done] worker=$i pid=$PID status=FAIL" | tee -a "$LOG_FILE"
    FAIL=1
  else
    echo "[worker-done] worker=$i pid=$PID status=ok" | tee -a "$LOG_FILE"
  fi
done
for WLOG in "${WORKER_LOGS[@]}"; do
  echo "----- BEGIN $WLOG -----" >> "$LOG_FILE"
  cat "$WLOG" >> "$LOG_FILE" || true
  echo "----- END   $WLOG -----" >> "$LOG_FILE"
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[FATAL] at least one eval worker failed; refusing to merge" \
    | tee -a "$LOG_FILE"
  exit 3
fi

MERGE_FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then MERGE_FORCE_FLAG="--force"; fi
echo "[$(date -Iseconds)] merging per-job results into leaderboards" \
  | tee -a "$LOG_FILE"
python scripts/merge_full_processbench_eval_results.py \
  --out_dir "$OUT_DIR" $MERGE_FORCE_FLAG 2>&1 | tee -a "$LOG_FILE"

echo "[merge] leaderboards:" | tee -a "$LOG_FILE"
ls -1 "$OUT_DIR"/leaderboard_full_pb_*.{csv,md} | tee -a "$LOG_FILE"

echo "[$(date -Iseconds)] full_pb_eval done" | tee -a "$LOG_FILE"
