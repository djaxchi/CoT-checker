#!/bin/bash
#SBATCH --job-name=conicity_cpu
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Class-cone geometry of the unified-harness dense_last representations.
# Pure numpy/sklearn on the already-encoded vector cache, no GPU needed.
# h.npy is memory-mapped, so the 3.7 GB probe_train_full never loads whole;
# peak RSS is driven by the capped fit subsample (--max_train x 3584 float32).

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/cache/qwen2_5_7b}"
TAG="${TAG:-dense_last_7b}"
OUT_ROOT="${OUT_ROOT:-$RUN_ROOT/runs/conicity}"
MAX_TRAIN="${MAX_TRAIN:-100000}"
MAX_GEOM="${MAX_GEOM:-50000}"
N_SHUFFLE="${N_SHUFFLE:-50}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MPLBACKEND=Agg

mkdir -p "$OUT_ROOT"
cd "$PROJECT_ROOT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy scikit-learn matplotlib

echo "===== CONICITY / CLASS-CONE GEOMETRY ====="
echo "cache=$CACHE_DIR tag=$TAG out=$OUT_ROOT"
python scripts/analysis/conicity_class_geometry.py \
  --cache_dir "$CACHE_DIR" \
  --train_stem probe_train_full --val_stem val_5k --test_stem test_2k \
  --tag "$TAG" --out_root "$OUT_ROOT" \
  --max_train "$MAX_TRAIN" --max_geom "$MAX_GEOM" --n_shuffle "$N_SHUFFLE"

echo "[$(date)] conicity_cpu done"
