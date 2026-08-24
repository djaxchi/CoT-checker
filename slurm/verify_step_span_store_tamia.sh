#!/bin/bash
#SBATCH --job-name=verify_span_store
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# Gate on deleting the 984G master token store: prove the compact step-span
# store is equivalent before anything is removed.
#
# Row mode compares, for every item, the retained rows byte for byte against the
# master's [pre_step_boundary_idx : n_tokens) slice. Identical bytes mean every
# readout agrees, including readouts not written yet, which is the property the
# harness actually needs: a benchmark that varies the representation is only
# meaningful if all representations come from one set of activations.
# Readout mode then derives both stores end to end, which also exercises the
# offset rewriting in the meta rather than only the bytes.
#
# Reads only the kept rows (~147 GB), not the full 984G. Exits non-zero on any
# mismatch, so the deletion must not be chained to a failed run.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
MASTER="${MASTER:-$SCRATCH/cot_mech/dense_full_7b_v1/repstore/tokens_last_layer}"
COMPACT="${COMPACT:-/project/aip-azouaq/$USER/cot_mech/dense_full_7b_v1/repstore/step_spans}"
SPLITS="${SPLITS:-probe_train_full}"
MODE="${MODE:-both}"
REPORT_EVERY="${REPORT_EVERY:-50000}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-verify_span_store}  job_id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
master     : $MASTER
compact    : $COMPACT
splits     : $SPLITS   mode: $MODE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy

python scripts/verify_step_span_store.py \
  --master "$MASTER" --compact "$COMPACT" \
  --splits $SPLITS --mode "$MODE" --report_every "$REPORT_EVERY"

echo "[$(date)] verification passed for: $SPLITS"
