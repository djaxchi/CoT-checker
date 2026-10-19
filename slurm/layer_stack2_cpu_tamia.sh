#!/bin/bash
#SBATCH --job-name=layer_stack2
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:30:00
#SBATCH --output=%x-%j.out
# layer_stack_v0 follow-up: the winning stack had C at the grid edge (1e-4).
# Extend C downward and add the full 6-layer x 2-readout stack. Not prespecified;
# reported as a robustness check of the first run.
set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
V="$SCRATCH/cot_mech/backbone_geometry_v0/vectors"
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32 LAYER_STACK_CS="1e-6,1e-5,3e-5,1e-4,3e-4"
cd "$HOME/CoT-checker"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index numpy scipy scikit-learn 2>&1 | tail -1
python scripts/analysis/layer_stack.py --dir "$V/q3_ver" --singles 35:last 30:last \
  --stacks 18:mean,24:mean,30:mean,35:mean,24:last,30:last,35:last \
           12:mean,18:mean,24:mean,30:mean,35:mean,36:mean,12:last,18:last,24:last,30:last,35:last,36:last \
  --out "$SCRATCH/cot_mech/backbone_geometry_v0/layer_stack_q3_cgrid.json"
