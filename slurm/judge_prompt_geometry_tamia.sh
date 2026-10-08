#!/bin/bash
#SBATCH --job-name=jp_geo
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00

# judge_prompt_v1 learner-free geometry (closed-form LDA, QDA, PCs; no kNN), one
# readout per GPU: span last_token / step_mean / boundary_stats (pre-derived
# vectors) and the verdict token. Comparable with prm_geometry_v1's rows.
set -euo pipefail
: "${RUN:?}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD)"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch numpy 2>&1 | tail -1
GEO="$RUN/geometry"; mkdir -p "$GEO" "$RUN/logs"
pids=(); g=0
for rep in last_token step_mean boundary_stats; do
  CUDA_VISIBLE_DEVICES=$g python scripts/analysis/prm_geometry.py --out_dir "$GEO" --no_knn \
    --prederived --backbone "judge_span=$RUN/vec/span,$RUN/vec/pb_span" --reps "$rep" \
    > "$RUN/logs/geometry_span_${rep}-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!); g=$((g + 1))
done
CUDA_VISIBLE_DEVICES=3 python scripts/analysis/prm_geometry.py --out_dir "$GEO" --no_knn \
  --backbone "judge_verdict=$RUN/judge_token,$RUN/pb_judge_token" --reps last_token \
  > "$RUN/logs/geometry_verdict-${SLURM_JOB_ID}.log" 2>&1 &
pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -5 "$RUN"/logs/geometry_*-"${SLURM_JOB_ID}".log
(( fail == 0 )) || { echo "[FATAL] geometry failed" >&2; exit 1; }
echo "[$(date)] geometry done"
