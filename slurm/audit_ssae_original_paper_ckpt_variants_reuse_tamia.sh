#!/bin/bash
# Reuse-only audit for original-paper Qwen2.5-0.5B SSAE checkpoint latents.
#
# This job does not download, convert, or encode anything. It trains lightweight
# probe variants on the existing PRM800K SSAE latents, evaluates them on the
# existing full ProcessBench SSAE latents, and writes separate leaderboards.

#SBATCH --job-name=ssae_paper_var_reuse
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load StdEnv/2023 python/3.12

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/prestudy_v1}"
RUNS_DIR="$RUN_ROOT/runs"
SOURCE_METHOD="ssae_original_paper_ckpt_qwen0p5b"
SOURCE_RUN_DIR="$RUNS_DIR/$SOURCE_METHOD"
SOURCE_LATENTS_DIR="$SOURCE_RUN_DIR/latents"
SOURCE_PB_ROOT="$SOURCE_RUN_DIR/latents_full_pb"
VARIANT_RUNS_ROOT="$RUNS_DIR/ssae_original_paper_ckpt_qwen0p5b_variants"
OUT_DIR="$RUNS_DIR/full_processbench_eval_audit_original_paper_ckpt_qwen0p5b_variants"
LOG_DIR="$RUN_ROOT/logs"
NUM_WORKERS="${NUM_WORKERS:-1}"
ORACLE_STEP="${ORACLE_STEP:-0.005}"
FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_FLAG="--force"; fi

METHODS=(
  ssae_original_paper_ckpt_qwen0p5b_positive
  ssae_original_paper_ckpt_qwen0p5b_contrastive
)
VARIANTS=(positive contrastive)

mkdir -p "$VARIANT_RUNS_ROOT" "$OUT_DIR" "$LOG_DIR"

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
JID="${SLURM_JOB_ID:-$$}"
LOG_FILE="$LOG_DIR/ssae_paper_variants_reuse-${JID}.log"

cat <<BANNER | tee -a "$LOG_FILE"
================================================================
job          : ${SLURM_JOB_NAME:-ssae_paper_var_reuse}
job_id       : ${JID}
hostname     : $(hostname)
date         : $(date -Iseconds)
git_commit   : $GIT_COMMIT
source_method: $SOURCE_METHOD
source_lat   : $SOURCE_LATENTS_DIR
source_pb    : $SOURCE_PB_ROOT
variant_runs : $VARIANT_RUNS_ROOT
out_dir      : $OUT_DIR
num_workers  : $NUM_WORKERS
oracle_step  : $ORACLE_STEP
log_file     : $LOG_FILE
note         : reuse-only probe/eval variants over one fixed Qwen2.5-0.5B
               original paper SSAE checkpoint; no representation retraining.
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

echo "[$(date -Iseconds)] sanity-checking reusable latent artifacts" | tee -a "$LOG_FILE"
python - <<PY 2>&1 | tee -a "$LOG_FILE"
import json
from pathlib import Path
import numpy as np

lat = Path("${SOURCE_LATENTS_DIR}")
pb = Path("${SOURCE_PB_ROOT}")
expected_pb = {
    "gsm8k": 2082,
    "math": 6505,
    "olympiadbench": 8819,
    "omnimath": 8291,
    "combined": 25697,
}

def meta_rows(path: Path) -> int:
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())

def check_pair(stem: str, expected_rows: int | None = None) -> None:
    z = np.load(lat / f"{stem}_z.npy")
    y = np.load(lat / f"{stem}_y.npy")
    n_meta = meta_rows(lat / f"{stem}_meta.jsonl")
    if z.ndim != 2 or z.shape[1] != 896:
        raise SystemExit(f"{stem}: expected z.shape[1] == 896, got {z.shape}")
    if z.shape[0] != y.shape[0] or z.shape[0] != n_meta:
        raise SystemExit(f"{stem}: row mismatch z={z.shape[0]} y={y.shape[0]} meta={n_meta}")
    if expected_rows is not None and z.shape[0] != expected_rows:
        raise SystemExit(f"{stem}: expected {expected_rows} rows, got {z.shape[0]}")
    if not np.all(np.isfinite(z)):
        raise SystemExit(f"{stem}: non-finite z values")
    print(f"[check] {stem}: z={z.shape} y={y.shape} meta={n_meta} finite=True")

check_pair("probe_train_40k", 40000)
check_pair("val_1k", 1000)

for subset, expected in expected_pb.items():
    z_path = pb / subset / "pb_step_z.npy"
    m_path = pb / subset / "pb_step_meta.jsonl"
    z = np.load(z_path)
    n_meta = meta_rows(m_path)
    if z.ndim != 2 or z.shape[1] != 896:
        raise SystemExit(f"{subset}: expected z.shape[1] == 896, got {z.shape}")
    if z.shape[0] != expected or n_meta != expected:
        raise SystemExit(f"{subset}: expected {expected} rows, got z={z.shape[0]} meta={n_meta}")
    if not np.all(np.isfinite(z)):
        raise SystemExit(f"{subset}: non-finite z values")
    print(f"[check] pb/{subset}: z={z.shape} meta={n_meta} finite=True")
PY

echo "[$(date -Iseconds)] training probe-only variants" | tee -a "$LOG_FILE"
for I in "${!METHODS[@]}"; do
  METHOD="${METHODS[$I]}"
  VARIANT="${VARIANTS[$I]}"
  RUN_DIR="$VARIANT_RUNS_ROOT/$METHOD"
  mkdir -p "$RUN_DIR"
  echo "[probe] method=$METHOD variant=$VARIANT run_dir=$RUN_DIR" | tee -a "$LOG_FILE"
  python scripts/train_ssae_probe_variant_reuse.py \
    --method "$METHOD" \
    --variant "$VARIANT" \
    --latents_dir "$SOURCE_LATENTS_DIR" \
    --out_dir "$RUN_DIR" \
    --expected_dim 896 \
    --seed 42 \
    --epochs_probe 50 \
    --batch_size 512 \
    --lr_probe 1e-3 2>&1 | tee -a "$LOG_FILE"
done

JOBS_DIR="$OUT_DIR/_jobs"
mkdir -p "$JOBS_DIR"
SHARD_PREFIX="$JOBS_DIR/eval_jobs"

echo "[$(date -Iseconds)] building full-PB eval jobs" | tee -a "$LOG_FILE"
python scripts/build_full_processbench_eval_jobs.py \
  --runs_root "$VARIANT_RUNS_ROOT" \
  --dense_pb_cache_root "$SOURCE_PB_ROOT" \
  --ssae_pb_root_override "$SOURCE_PB_ROOT" \
  --out_dir "$OUT_DIR" \
  --methods "${METHODS[@]}" \
  --include_combined \
  --out_jobs_jsonl "$JOBS_DIR/eval_jobs.jsonl" \
  --num_workers "$NUM_WORKERS" \
  --shard_prefix "$SHARD_PREFIX" 2>&1 | tee -a "$LOG_FILE"

TOTAL_JOBS=$(wc -l < "$JOBS_DIR/eval_jobs.jsonl" | awk '{print $1}')
echo "[plan] total_jobs=$TOTAL_JOBS workers=$NUM_WORKERS" | tee -a "$LOG_FILE"
if [[ "$TOTAL_JOBS" -ne 10 ]]; then
  echo "[FATAL] expected 10 eval jobs (2 methods x 5 subsets), got $TOTAL_JOBS" \
    | tee -a "$LOG_FILE"
  exit 2
fi

WORKER_PIDS=()
WORKER_LOGS=()
for ((W=0; W<NUM_WORKERS; W++)); do
  GPU=$W
  JOBS_JSON="${SHARD_PREFIX}_worker_${W}.json"
  WLOG="$LOG_DIR/ssae_paper_variants_reuse-${JID}-worker${W}.log"
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
      $FORCE_FLAG
    echo "[worker] worker_id=$W rc=$? end=$(date -Iseconds)"
  ) >"$WLOG" 2>&1 &
  WORKER_PIDS+=("$!")
done

FAIL=0
for I in "${!WORKER_PIDS[@]}"; do
  PID=${WORKER_PIDS[$I]}
  if ! wait "$PID"; then
    echo "[worker-done] worker=$I pid=$PID status=FAIL" | tee -a "$LOG_FILE"
    FAIL=1
  else
    echo "[worker-done] worker=$I pid=$PID status=ok" | tee -a "$LOG_FILE"
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

echo "[$(date -Iseconds)] merging per-job results into leaderboards" \
  | tee -a "$LOG_FILE"
python scripts/merge_full_processbench_eval_results.py \
  --out_dir "$OUT_DIR" \
  --methods "${METHODS[@]}" \
  $FORCE_FLAG 2>&1 | tee -a "$LOG_FILE"

echo "[leaderboards]" | tee -a "$LOG_FILE"
cat "$OUT_DIR/leaderboard_full_pb_method_averages.md" | tee -a "$LOG_FILE"
cat "$OUT_DIR/leaderboard_full_pb_val_threshold.md" | tee -a "$LOG_FILE"
cat "$OUT_DIR/leaderboard_full_pb_oracle_threshold.md" | tee -a "$LOG_FILE"

echo "[$(date -Iseconds)] reuse variant audit done" | tee -a "$LOG_FILE"
