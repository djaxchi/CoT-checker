#!/bin/bash
#SBATCH --job-name=geomctl
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Is the geometry worth anything, or is it the token count again?
#
# dir_geom scores 0.7757 against dir's 0.7522, a gain of 0.0235. But geom carries
# log token count, geom WITHOUT it scores 0.5182, and step length alone scores
# 0.7039. Within length strata the gain shrinks to 0.0091, which is about what
# bolting bare length onto a representation was already worth (mean_pluslen
# gained 0.0080 there).
#
# So the gain has two candidate explanations and the table cannot separate them.
# These two controls can:
#
#   dir_geom_nolen   the same 20 geometry numbers with length removed. If this
#                    ties dir, the geometry contributes nothing and dir_geom is
#                    a length feature in an expensive wrapper.
#   dir_pluslen      dir with log length and nothing else. If this matches
#                    dir_geom, same conclusion from the other direction.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
POOL="${POOL:-$RUN_ROOT/poolings}"
REL="${REL:-$RUN_ROOT/relational}"
CTL="${CTL:-$RUN_ROOT/geom_control}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$CTL"
cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy

# stack_layers refuses unless the rows are identical, which also verifies that
# the poolings and relational passes sampled the same steps in the same order
python scripts/stack_layers.py --npz "$POOL/dir.npz" "$REL/geom_nolen.npz" \
  --out "$CTL/dir_geom_nolen.npz"
python scripts/stack_layers.py --npz "$POOL/dir.npz" "$POOL/surface_length.npz" \
  --out "$CTL/dir_pluslen.npz"
python scripts/stack_layers.py --npz "$POOL/mean_residual.npz" "$REL/geom_nolen.npz" \
  --out "$CTL/mean_residual_geom_nolen.npz"
python scripts/stack_layers.py --npz "$POOL/dir.npz" "$REL/layer_angle.npz" \
  --out "$CTL/dir_layer_angle.npz"

python scripts/ridge_screen.py --npz "$CTL"/*.npz \
  "$POOL/dir.npz" "$POOL/mean_residual.npz" "$REL/geom_nolen.npz" \
  "$REL/dir_geom.npz" "$POOL/surface_length.npz" \
  --out "$RUN_ROOT/geom_control.json"
echo "[$(date)] geomctl done"
