#!/bin/bash
#SBATCH --job-name=sae_loc
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:30:00
#SBATCH --output=%x-%j.out

# sae_localization_v0 (docs/sae_localization_v0_plan.md): CPU-only analyses on
# vectors already on scratch. WHAT=loc (SAE top-k ladder), oc (one-class) or strata (step-type strata).
set -euo pipefail
: "${WHAT:?loc, oc or strata}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
BASE="${BASE:-$SCRATCH/cot_mech/qwen3_8b_v1}"
OUT="${OUT:-$BASE/runs/sae_localization_v0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) what=$WHAT"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index numpy scipy scikit-learn 2>&1 | tail -1
if [[ "$WHAT" == loc ]]; then
  python scripts/analysis/sae_feature_localization.py --sae_dir "$BASE/sae_reps" \
    --dense_cache "$BASE/cache/grid_vectors" --out "$OUT/loc"
elif [[ "$WHAT" == strata ]]; then
  python scripts/analysis/step_type_strata.py --sae_dir "$BASE/sae_reps" \
    --dense_cache "$BASE/cache/grid_vectors" --top_features "$OUT/loc/top_features.json" \
    --out "$OUT/strata/summary.json"
else
  python scripts/analysis/one_class_correct.py --dense_cache "$BASE/cache/grid_vectors" \
    --out "$OUT/oc"
fi
