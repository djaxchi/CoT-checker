#!/bin/bash
#SBATCH --job-name=explaingeom
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Which of the 20 geometry features carry the gain?
#
# lengthfree_geom beats a dimension-matched step_mean by 0.026 F1 at calib-20.
# The project's mission is to explain the signal rather than move the score, and
# the current explanation is a 20-dimensional shrug: the features were designed
# together, score 0.5182 alone, and help only in combination.
#
# Add-one-in says what a feature is worth when nothing can stand in for it.
# Leave-one-out says what is lost that nothing else replaces. The pair separates
# load-bearing from redundant from dead weight.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
CTL="${CTL:-$RUN_ROOT/geom_control}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

python scripts/explain_geom.py --npz "$CTL/mean_residual_geom_nolen.npz" \
  --out "$RUN_ROOT/explain_geom.json"
echo "[$(date)] explaingeom done"
