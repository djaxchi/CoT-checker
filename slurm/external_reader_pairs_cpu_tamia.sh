#!/bin/bash
#SBATCH --job-name=ext_pairs
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
cd "$HOME/CoT-checker"
virtualenv --no-download "$SLURM_TMPDIR/env" >/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index numpy scipy scikit-learn sympy 2>&1 | tail -1
python scripts/analysis/external_reader_pairs.py --tts_root "${TTS_ROOT:?}" \
  --vectors "$SCRATCH/cot_mech/external_reader_v0/vectors" \
  --train_vectors "$SCRATCH/cot_mech/backbone_geometry_v0/vectors" \
  --base_grid "$SCRATCH/cot_mech/qwen3_8b_v1/cache/grid_vectors" \
  --out "$SCRATCH/cot_mech/external_reader_v0/pairs.json"
