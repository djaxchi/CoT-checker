#!/bin/bash
#SBATCH --job-name=instruct_leaderboard
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

# Reuse the fixed trainer in an isolated source snapshot. No backbone encoding.
set -euo pipefail
: "${PROJECT_ROOT:?Set the isolated code snapshot}"
: "${RUN_ROOT:?Set the dedicated leaderboard output root}"
: "${GROUP:?Set vectors or sequences}"
case "$GROUP" in vectors|sequences) ;; *) exit 2 ;; esac
export PRM_STORE="/scratch/d/dchikhi/cot_mech/qwen3_8b_instruct_v1/repstore/step_spans"
export PB_STORE="/scratch/d/dchikhi/cot_mech/qwen3_8b_instruct_v1/repstore/pb_step_spans"
export OUT_ROOT="$RUN_ROOT/cells"
# Keep derived caches on node-local storage. Shared scratch has little headroom
# and concurrent Instruct generation jobs are using it.
export VEC_CACHE="$SLURM_TMPDIR/instruct_grid_vectors"
export CELLS_FILE="$PROJECT_ROOT/experiments/instruct_leaderboard_v1/$GROUP.cells"
export RESCALE=none SEEDS="42 43 44" EPOCHS=30 BATCH_SIZE=256 HP_SEARCH_CAP=100000
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export N_GPUS=4 PRELOAD_BUDGET_GB=60
if [[ "$GROUP" == sequences ]]; then
  # Two 163 GB preloads fit on the 500 GB node; four do not.
  export N_GPUS=2 PRELOAD_BUDGET_GB=185
fi
module purge
bash "$PROJECT_ROOT/slurm/train_rep_grid_7b_tamia.sh"
# The shared trainer can finish after some cells fail. Validate the requested
# roster before allowing dependent jobs to see a successful Slurm exit.
source "$SLURM_TMPDIR/env/bin/activate"
python "$PROJECT_ROOT/scripts/validate_instruct_leaderboard.py" \
  --root "$OUT_ROOT" --cells-file "$CELLS_FILE" --reference "$RUN_ROOT/reference.json"
