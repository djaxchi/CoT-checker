#!/bin/bash
#SBATCH --job-name=resgeom
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Push on the one combination that worked, and price it.
#
# mean_residual (length regressed out of every position) scores 0.7282 alone, the
# worst of the dense representations. geom_nolen (20 scale-free geometry numbers,
# length excluded) scores 0.5182 alone, barely above chance. Together they score
# 0.7897, the best result in the search, and 0.7377 within length strata, also the
# best. The combination is worth far more than either part.
#
# Three things this settles:
#   - is 0.7897 over dir_geom's 0.7757 real? paired bootstrap on identical rows
#   - does it hold on layer 26, which beats layer 35 alone at every budget
#   - is removing length then re-supplying shape better than just keeping length?
#     mean_geom_nolen and mean_residual_pluslen are the two directions of that

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
POOL="${POOL:-$RUN_ROOT/poolings}"
POOL26="${POOL26:-$RUN_ROOT/poolings_L26}"
REL="${REL:-$RUN_ROOT/relational}"
CTL="${CTL:-$RUN_ROOT/geom_control}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$CTL"
cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

# layer 26 versions of the two ingredients
for r in mean dir; do
  python scripts/residualize_length.py --npz "$POOL26/$r.npz" --mode residual \
    --out "$CTL/${r}_L26_residual.npz"
  python scripts/stack_layers.py --npz "$CTL/${r}_L26_residual.npz" "$REL/geom_nolen.npz" \
    --out "$CTL/${r}_L26_residual_geom_nolen.npz"
done

# does the recipe need the residualisation, or would plain content plus geometry do
python scripts/stack_layers.py --npz "$POOL/mean.npz" "$REL/geom_nolen.npz" \
  --out "$CTL/mean_geom_nolen.npz"
# and is removing length then re-supplying shape better than just keeping length
python scripts/stack_layers.py --npz "$POOL/mean_residual.npz" "$POOL/surface_length.npz" \
  --out "$CTL/mean_residual_pluslen.npz"
python scripts/residualize_length.py --npz "$POOL/dir.npz" --mode residual \
  --out "$CTL/dir_residual.npz"
python scripts/stack_layers.py --npz "$CTL/dir_residual.npz" "$REL/geom_nolen.npz" \
  --out "$CTL/dir_residual_geom_nolen.npz"

python scripts/ridge_screen.py \
  --npz "$CTL"/*.npz "$POOL/mean_residual.npz" "$POOL/dir.npz" "$POOL/mean.npz" \
        "$REL/dir_geom.npz" "$REL/geom_nolen.npz" "$POOL/surface_length.npz" \
  --bootstrap 1000 --ref mean_residual_geom_nolen \
  --out "$RUN_ROOT/residual_geom.json"
echo "[$(date)] resgeom done"
