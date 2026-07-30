#!/bin/bash
#SBATCH --job-name=lookahead_cpu
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# CPU-only within-PB ceiling test on the already-built full-solution store
# (encode_pb_full_store_7b did the GPU encode). Pure numpy, no GPU needed.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
REP_ROOT="$RUN_ROOT/repstore/pb_full_solution"
SUBSETS="${SUBSETS:-gsm8k math olympiadbench omnimath}"
WINDOWS="${WINDOWS:-1 2 -1}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$PROJECT_ROOT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy

echo "===== LOOKAHEAD CEILING (within-PB group CV) ====="
python scripts/analysis/lookahead_ceiling.py \
  --store_root "$REP_ROOT" --subsets $SUBSETS --windows $WINDOWS

echo "[$(date)] lookahead_cpu done"
