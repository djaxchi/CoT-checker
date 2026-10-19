#!/bin/bash
#SBATCH --job-name=xmodel_dir
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
V="$SCRATCH/cot_mech/backbone_geometry_v0/vectors"
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
cd "$HOME/CoT-checker"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index numpy scipy scikit-learn 2>&1 | tail -1
python scripts/analysis/cross_model_direction.py --prm "$V/prm_ver" --parent "$V/parent_ver" \
  --out "$SCRATCH/cot_mech/backbone_geometry_v0/cross_model_direction.json"
