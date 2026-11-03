#!/bin/bash
#SBATCH --job-name=instep_v1_final
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# InStep paper V1, run C1 (docs/instep_v1_runbook.md): CPU-only final aggregation. Every arm
# that exists under $OUT, reprobe_cmp_v1 and reprobe_cmp_v2 is scored on the three backbones
# (analyze.py metrics, paired trace bootstraps, ProcessBench threshold-free metrics and
# calib-20, Tier B paper rows + consistency check, best-of-N, abstention), then tables and
# figures. Missing arms are skipped and listed in results.md ("What is missing").
# Outputs: $FINAL/{results.md,tables.tex,final.json,fig*.png,fig*.pdf}.
#   SMOKE=1  writes to $OUT/final_smoke with 200 resamples; by default it stands the old
#            read-token scores (reprobe_cmp_v1/runs/rt/rt_ds_seed{0,1,2}) in for InStep-DS so
#            the whole pipeline runs before the E1 scores exist. Submit it with --time=00:20:00.
#   OVERRIDE="bb:key=dir1,dir2 ..."  replaces arms' seed dirs (see paper_v1.py --override).
set -euo pipefail
SMOKE="${SMOKE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
CM="/project/aip-azouaq/$USER/cot_mech"
OUT="${OUT:-$CM/instep_v1}"
V1="${V1:-$CM/reprobe_cmp_v1}"
V2="${V2:-$CM/reprobe_cmp_v2}"
if [[ $SMOKE == 0 ]]; then
  FINAL="${FINAL:-$OUT/final}"; N_BOOT="${N_BOOT:-2000}"; OVERRIDE="${OVERRIDE:-}"
else
  FINAL="${FINAL:-$OUT/final_smoke}"; N_BOOT="${N_BOOT:-200}"
  OVERRIDE="${OVERRIDE-qwen3_8b:instep_ds=$V1/runs/rt/rt_ds_seed0,$V1/runs/rt/rt_ds_seed1,$V1/runs/rt/rt_ds_seed2}"
fi
mkdir -p "$FINAL"
module purge
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
# one process per bootstrap task; keep BLAS single-threaded inside each
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD 2>/dev/null || echo none) smoke=$SMOKE out=$OUT final=$FINAL n_boot=$N_BOOT"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index numpy scipy scikit-learn matplotlib 2>&1 | tail -1
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl"
OV=(); [[ -n "$OVERRIDE" ]] && read -r -a OV <<< "$OVERRIDE"
python scripts/instep_v1/paper_v1.py --out_root "$OUT" --v1 "$V1" --v2 "$V2" --final "$FINAL" \
  --n_boot "$N_BOOT" --pb_boot "$N_BOOT" --bon_boot "$N_BOOT" --workers "${SLURM_CPUS_PER_TASK:-48}" \
  ${OV[@]+--override "${OV[@]}"} 2>&1 | tee "$FINAL/run-${SLURM_JOB_ID:-local}.log"
ls -la "$FINAL"
