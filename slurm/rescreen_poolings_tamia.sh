#!/bin/bash
#SBATCH --job-name=rescreen
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out

# Re-screen every cached pooling with the standardised screen.
#
# The previous pass reported the length-only baselines at 0.296 PB step AUROC,
# below chance, on data whose own statistics say longer steps are more often
# wrong in both domains. The screen used one learning rate for every
# representation, and a one-dimensional probe never moved off its random
# initialisation. Inputs are standardised now.
#
# The vectors themselves are unchanged and already on disk, so this only refits
# the probes. The three numbers that matter are surface_length, mean, and
# mean_pluslen, which together say whether the leaderboard is reading the
# reasoning or reading how long the step is.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
OUT_DIR="${OUT_DIR:-$SCRATCH/cot_mech/qwen3_8b_v1/poolings}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

python scripts/screen_representation.py --npz "$OUT_DIR"/*.npz \
  --out "$OUT_DIR/screen_standardised.json"
echo "[$(date)] rescreen done"
