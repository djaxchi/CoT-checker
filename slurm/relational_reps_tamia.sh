#!/bin/bash
#SBATCH --job-name=relreps
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Representations built from relations rather than from the step's own content.
#
# Every pooling so far is some average of the step's token states, and they land
# within 0.03 of each other inside length strata. Two things the store already
# holds have never been used: the pre-step boundary state, which `poolings()`
# takes as an argument and no pooling touches, and a second layer, which was
# encoded for a stacking test that turned out to be a wider vector rather than a
# better idea.
#
#   contribution    mean(step) - boundary. What the step ADDED, not the state it
#                   left behind. A step is wrong given what came before.
#   geom            about twenty numbers of pure geometry, no direction in model
#                   space at all. Tests the conicity finding directly, and does
#                   it without a metric, since an angle is already scale free.
#   layer_angle     twelve numbers of disagreement between layers 26 and 35. Not
#                   a concatenation: how much the late blocks REWROTE this step.
#
# All are low-dimensional, which also sidesteps the width-dependent convergence
# problem that made the stacking result unreadable.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/relational}"
POOL_DIR="${POOL_DIR:-$RUN_ROOT/poolings}"
N_TRAIN="${N_TRAIN:-60000}"
N_PB="${N_PB:-4000}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

python scripts/relational_reps.py \
  --prm_store   "$RUN_ROOT/repstore/step_spans" \
  --pb_store    "$RUN_ROOT/repstore/pb_step_spans" \
  --prm_store_b "$RUN_ROOT/repstore/step_spans_L26" \
  --pb_store_b  "$RUN_ROOT/repstore/pb_step_spans_L26" \
  --out_dir "$OUT_DIR" --n_train "$N_TRAIN" --n_pb "$N_PB"

# The low-dimensional representations need a budget that actually fits them; the
# stacking discrepancy showed the fixed budget is width dependent. Screen at the
# incumbent budget for comparability AND at a converged one.
REF="$POOL_DIR/dir.npz $POOL_DIR/mean.npz $POOL_DIR/surface_length.npz"
for cfg in "8 0.001 incumbent" "60 0.01 converged"; do
  set -- $cfg
  echo "=== screen: epochs $1 lr $2 ($3) ==="
  python scripts/screen_representation.py --npz "$OUT_DIR"/*.npz $REF \
    --epochs "$1" --lr "$2" --out "$OUT_DIR/screen_$3.json"
  echo "=== within length strata: epochs $1 lr $2 ($3) ==="
  python scripts/stratified_auroc.py --npz "$OUT_DIR"/*.npz $REF \
    --epochs "$1" --lr "$2" --n_bins 50 --out "$OUT_DIR/stratified_$3.json"
done
echo "[$(date)] relreps done"
