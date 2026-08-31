#!/bin/bash
#SBATCH --job-name=screen_pool
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Screen alternative ways of pooling a step's tokens.
#
# The grid found that which rows you pool dominates, and that compression of any
# kind costs (SAE caps ~0.06 below the dense ceiling; every bottleneck ties or
# loses to the raw representation). So the remaining room is the pooling rule,
# which has only ever been mean / max / min / std / last.
#
# The lead hypothesis is grounded in this project's own probe-anatomy result:
# the correctness signal is carried by DIRECTION, not magnitude. Plain mean lets
# a high-norm token dominate the average -- a single 50x-norm token moves it to
# cosine 0.551 -- while pooling L2-normalised tokens leaves it unmoved. If the
# finding is right, mean_l2 should beat mean.
#
# CPU only: sampling the store and pooling is numpy, and the screen itself is
# sub-second per representation.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/poolings}"
N_TRAIN="${N_TRAIN:-60000}"
N_PB="${N_PB:-4000}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

python scripts/screen_poolings.py \
  --prm_store "$RUN_ROOT/repstore/step_spans" \
  --pb_store "$RUN_ROOT/repstore/pb_step_spans" \
  --out_dir "$OUT_DIR" --n_train "$N_TRAIN" --n_pb "$N_PB"

# Length variants of the leading poolings. PRM800K steps average 38.8 tokens and
# ProcessBench steps run 56 to 94, so a representation that encodes length gives
# the probe a boundary fitted to the short domain and evaluated on the long one.
LEN_REPS="${LEN_REPS:-mean centered quantiles}"
for r in $LEN_REPS; do
  [ -f "$OUT_DIR/$r.npz" ] || continue
  for m in residual withlen; do
    python scripts/residualize_length.py --npz "$OUT_DIR/$r.npz" --mode "$m" \
      --out "$OUT_DIR/${r}_${m}.npz"
  done
done

echo
echo "=== screen ==="
python scripts/screen_representation.py --npz "$OUT_DIR"/*.npz --out "$OUT_DIR/screen.json"
echo "[$(date)] screen_pool done"
