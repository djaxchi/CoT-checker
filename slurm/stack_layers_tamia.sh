#!/bin/bash
#SBATCH --job-name=stacklayer
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# Read the leading poolings at layer 26 as well as 35, and stack them.
#
# The one large gain this project's earlier representation search ever found was
# reading more than one layer: about +0.05 AUC, where every compression scheme
# lost. Never retested on Qwen3, and the whole grid reads a single layer.
#
# Only the poolings that lead the within-length column are read, since a screen
# that costs a full pass over every pooling is not a screen. Alignment between
# the two layers is enforced inside stack_layers.py rather than assumed.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/poolings}"
L26_DIR="${L26_DIR:-$RUN_ROOT/poolings_L26}"
NAMES="${NAMES:-mean dir mean_l2 atypical}"
N_TRAIN="${N_TRAIN:-60000}"
N_PB="${N_PB:-4000}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$L26_DIR"
cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

# same seed as the layer-35 pass, so the sampled rows are the same steps
python scripts/screen_poolings.py \
  --prm_store "$RUN_ROOT/repstore/step_spans_L26" \
  --pb_store "$RUN_ROOT/repstore/pb_step_spans_L26" \
  --out_dir "$L26_DIR" --n_train "$N_TRAIN" --n_pb "$N_PB" --names $NAMES

STACKED=""
for n in $NAMES; do
  python scripts/stack_layers.py --npz "$L26_DIR/$n.npz" "$OUT_DIR/$n.npz" \
    --out "$OUT_DIR/${n}_L26L35.npz"
  # layer 26 on its own, so a stacking gain is separable from "26 is just better"
  cp "$L26_DIR/$n.npz" "$OUT_DIR/${n}_L26only.npz"
  STACKED="$STACKED $OUT_DIR/${n}_L26L35.npz $OUT_DIR/${n}_L26only.npz"
done

echo "=== screen ==="
python scripts/screen_representation.py --npz $STACKED "$OUT_DIR/mean.npz" \
  "$OUT_DIR/dir.npz" --out "$OUT_DIR/screen_stacked.json"
echo "=== within length strata ==="
python scripts/stratified_auroc.py --npz $STACKED "$OUT_DIR/mean.npz" \
  "$OUT_DIR/dir.npz" --n_bins 50 --out "$OUT_DIR/stratified_stacked.json"
echo "[$(date)] stacklayer done"
