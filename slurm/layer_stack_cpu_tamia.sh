#!/bin/bash
#SBATCH --job-name=layer_stack
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# layer_stack_v0 on the saved backbone_geometry_v0 vectors (CPU only).
set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
V="$SCRATCH/cot_mech/backbone_geometry_v0/vectors"
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
cd "$HOME/CoT-checker"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index numpy scipy scikit-learn 2>&1 | tail -1
python scripts/analysis/layer_stack.py --dir "$V/q3_ver" \
  --singles 24:mean 30:last 35:last 35:mean 36:mean \
  --stacks 35:last,35:mean 24:mean,30:mean,35:mean,36:mean 24:last,30:last,35:last,36:last \
           18:mean,24:mean,30:mean,35:mean,24:last,30:last,35:last \
  --out "$SCRATCH/cot_mech/backbone_geometry_v0/layer_stack_q3.json"
