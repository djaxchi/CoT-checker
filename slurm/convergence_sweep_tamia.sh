#!/bin/bash
#SBATCH --job-name=converge
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Is the screen ranking representations, or ranking how fast they fit?
#
# screen_representation.py and stratified_auroc.py fit the same probe on the same
# file and disagreed by 0.021 on dir_L26L35 (8,192 dim) while agreeing to 0.004 on
# dir (4,096 dim). The only difference was 50,000 vs 60,000 training rows. A gap
# that grows with width is what an undertrained probe looks like.
#
# This is decisive for the current hypothesis: stacking two layers doubles the
# width, so "stacking wins" and "the stack is at a different point on its
# optimisation path" are indistinguishable until the budget is swept.

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

REPS="dir dir_L26only dir_L26L35 mean mean_L26only mean_L26L35 quantiles first_last"
NPZ=""; for r in $REPS; do NPZ="$NPZ $OUT_DIR/$r.npz"; done

python scripts/convergence_sweep.py --npz $NPZ \
  --epochs 8 25 60 --lrs 0.001 0.01 --out "$OUT_DIR/convergence.json"

echo "=== re-screen the stack at whichever budget converged, plus strata ==="
python scripts/stratified_auroc.py --npz $NPZ --n_bins 50 --epochs 60 --lr 0.01 \
  --out "$OUT_DIR/stratified_converged.json"
echo "[$(date)] converge done"
