#!/bin/bash
#SBATCH --job-name=suppressor
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Is the geometry block a suppressor variable?
#
# The ablation left a puzzle. The 20 features are worth +0.0614 on top of
# length-free content, and any one of about eight recovers most of it on its own
# (cone_tightness_ratio +0.0458, cone_cos_mean +0.0457). Yet the block scores
# 0.5182 alone and 0.4675 inside length strata, below chance.
#
# A feature that carries nothing alone and a lot in combination is a suppressor:
# weakly related to the label, strongly related to the other predictors, earning
# its place by removing variance from them that the label does not explain. The
# prediction is falsifiable: near-zero correlation with the label, clear
# correlation with the content probe's score, and a partial correlation given
# that score that is clearly larger than the raw one.
#
# The correlation with log length is reported alongside, so "cone tightness is
# just a nonlinear stand-in for step length" rests on a number rather than on the
# inference from the block scoring 0.5182 where length scores 0.7039.

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

python scripts/suppressor_check.py --npz "$CTL/mean_residual_geom_nolen.npz" \
  --out "$RUN_ROOT/suppressor.json"
echo "[$(date)] suppressor done"
