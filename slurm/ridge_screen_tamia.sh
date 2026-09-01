#!/bin/bash
#SBATCH --job-name=ridge
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Rank representations with a probe that has no arbitrary budget in it.
#
# The convergence sweep found the screen's verdict was mostly about its budget:
# training longer made transfer monotonically worse for every representation, and
# the ranking at 8 epochs anti-correlated with the ranking at 60 (Spearman -0.07
# to -0.24). Ridge has a closed form, no epochs, no seed, and one knob whose two
# ends are the whitened and centroid rules the conicity study compared.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
POOL_DIR="${POOL_DIR:-$RUN_ROOT/poolings}"
REL_DIR="${REL_DIR:-$RUN_ROOT/relational}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

NPZ=""
for r in dir mean mean_l2 atypical dir_L26only mean_L26only mean_residual surface_length; do
  [ -f "$POOL_DIR/$r.npz" ] && NPZ="$NPZ $POOL_DIR/$r.npz"
done
for f in "$REL_DIR"/*.npz; do [ -f "$f" ] && NPZ="$NPZ $f"; done

python scripts/ridge_screen.py --npz $NPZ --out "$RUN_ROOT/ridge_screen.json"
echo "[$(date)] ridge done"
