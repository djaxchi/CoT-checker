#!/bin/bash
#SBATCH --job-name=step_span_store
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

# Compact the master token store to the rows any representation actually reads:
# the pre-step boundary state plus the step's own tokens. Measured over the
# 513,810 PRM800K train steps the span is 38.8 tokens against 283 for the full
# sequence, so this keeps 13.7% of the rows: 984G -> ~137GiB for the train split,
# 93G -> ~13G for ProcessBench.
#
# Two reasons to do this before the representation x learner grid:
#   1. it is what makes training a sequence learner on the FULL 513,810 steps
#      affordable, instead of the 150k subsample the v1 rows used;
#   2. $SCRATCH is over quota (1181GiB against 1024GiB) because the master store
#      dominates it, so the compact store goes to $STORE, and the master can be
#      dropped once the grid has been validated against it.
#
# Pure numpy over memory-mapped shards: no GPU, no model, low RSS (rows are
# copied item by item into an open_memmap output).
#
#   STORE_ROOT: master token store rep dir     SPLITS: split stems to compact
#   OUT_ROOT:   destination rep dir
# ProcessBench: STORE_ROOT=.../pb_tokens_last_layer SPLITS="gsm8k math olympiadbench omnimath"

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
STORE_ROOT="${STORE_ROOT:-$RUN_ROOT/repstore/tokens_last_layer}"
# Default to $STORE: data_setup.md's policy is master store in $SCRATCH (it is
# regenerable), derived data in $STORE, and $SCRATCH is the tier that is full.
OUT_ROOT="${OUT_ROOT:-$STORE/cot_mech/dense_full_7b_v1/repstore/step_spans}"
SPLITS="${SPLITS:-probe_train_full val_5k test_2k}"
NAME="${NAME:-step_spans}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
mkdir -p "$OUT_ROOT"

cat <<BANNER
================================================================
job         : ${SLURM_JOB_NAME:-step_span_store}  job_id: ${SLURM_JOB_ID:-N/A}
git_commit  : $GIT_COMMIT
store_root  : $STORE_ROOT
out_root    : $OUT_ROOT
splits      : $SPLITS
================================================================
BANNER
echo "[df] destination filesystem before writing:"
df -h "$OUT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy

python scripts/build_step_span_store.py \
  --store_root "$STORE_ROOT" \
  --out_root "$OUT_ROOT" \
  --splits $SPLITS \
  --name "$NAME"

echo "[df] destination filesystem after writing:"
df -h "$OUT_ROOT"
du -sh "$OUT_ROOT"/* || true
echo "[$(date)] step_span_store done"
