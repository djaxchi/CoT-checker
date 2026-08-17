#!/bin/bash
#SBATCH --job-name=lookahead_cpu
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

# CPU-only within-PB ceiling test on the already-built full-solution store
# (encode_pb_full_store_7b did the GPU encode). Pure numpy, no GPU needed.
#
# ONE SUBSET PER JOB. The unsharded version was killed by walltime four times in
# a row and never reached olympiadbench/omnimath, so submit_lookahead_shards.sh
# fans this out with SUBSETS set per job and a per-subset --time. Results are
# written to $OUT_JSON after every finished row, so even a killed job keeps the
# rows it completed.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
REP_ROOT="$RUN_ROOT/repstore/pb_full_solution"
SUBSETS="${SUBSETS:-gsm8k}"
WINDOWS="${WINDOWS:-1 -1}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/runs/lookahead}"
OUT_JSON="${OUT_JSON:-$OUT_DIR/$(echo $SUBSETS | tr ' ' '_').json}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy

echo "===== LOOKAHEAD CEILING (within-PB group CV) ====="
echo "subsets=$SUBSETS windows=$WINDOWS out=$OUT_JSON"
python scripts/analysis/lookahead_ceiling.py \
  --store_root "$REP_ROOT" --subsets $SUBSETS --windows $WINDOWS \
  --out_json "$OUT_JSON"

echo "[$(date)] lookahead_cpu done ($SUBSETS)"
