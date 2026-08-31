#!/bin/bash
#SBATCH --job-name=stratlen
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out

# Re-rank every cached pooling inside length strata.
#
# Step length alone scores 0.7039 on ProcessBench and the best representation
# scores 0.7700, so a token count answers most of the transfer benchmark. Inside
# an equal-length bin length carries nothing, so whatever separation survives is
# the representation's own. The length control is printed last and has to sit at
# about 0.5 for the rows above it to mean anything.

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

for K in 10 50; do
  echo "=== $K length bins ==="
  python scripts/stratified_auroc.py --npz "$OUT_DIR"/*.npz --n_bins "$K" \
    --out "$OUT_DIR/stratified_${K}.json"
done
echo "[$(date)] stratlen done"
