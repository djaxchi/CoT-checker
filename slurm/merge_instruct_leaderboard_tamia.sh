#!/bin/bash
#SBATCH --job-name=instruct_leaderboard_merge
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
: "${PROJECT_ROOT:?}" "${RUN_ROOT:?}"
module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.12
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index numpy
cd "$PROJECT_ROOT"
python scripts/validate_instruct_leaderboard.py --root "$RUN_ROOT/cells" \
  --cells-file experiments/instruct_leaderboard_v1/all.cells \
  --reference "$RUN_ROOT/reference.json" --ranked-out "$RUN_ROOT/ranked_leaderboard.md"
python scripts/merge_rep_grid_leaderboard.py --run_root "$RUN_ROOT/cells" \
  --out "$RUN_ROOT/leaderboard.md"
