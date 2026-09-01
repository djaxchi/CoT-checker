#!/bin/bash
#SBATCH --job-name=belief
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Read the stored states through the unembedding instead of around it.
#
# Every representation so far treats the residual stream as a vector to be
# pooled. None has been pushed through the unembedding, the one place the state
# becomes a statement about what the model thinks comes next. Two shifts:
# boundary against step end, and layer 26 against layer 35 at the same position.
#
# Needs a GPU only for the 151k-wide unembedding matmul; TamIA allocates H100
# nodes whole, so one GPU is used and three sit idle. Under two hours.
#
# The weights are read from the local snapshot only. Compute nodes have no
# internet, so a missing snapshot must fail here rather than hang on a download.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0 cuda/12.2

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_v1}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/belief}"
POOL="${POOL:-$RUN_ROOT/poolings}"
REL="${REL:-$RUN_ROOT/relational}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"

export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

SNAP="$HF_CACHE/hub/models--$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|--|g')/snapshots"
[ -d "$SNAP" ] || { echo "[FATAL] no local snapshot under $SNAP" >&2
                    echo "        on the LOGIN node: HF_HOME=$HF_CACHE hf download $MODEL_NAME_OR_PATH" >&2
                    exit 2; }

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy transformers accelerate

python scripts/belief_shift_reps.py \
  --prm_store   "$RUN_ROOT/repstore/step_spans" \
  --pb_store    "$RUN_ROOT/repstore/pb_step_spans" \
  --prm_store_b "$RUN_ROOT/repstore/step_spans_L26" \
  --pb_store_b  "$RUN_ROOT/repstore/pb_step_spans_L26" \
  --out_dir "$OUT_DIR" --hf_cache "$HF_CACHE" \
  --model_id "$MODEL_NAME_OR_PATH"

# Against the winner, and combined with it. `belief` is 11 numbers, so if it
# adds anything it will add the way geom_nolen did: useless alone, complementary.
python scripts/stack_layers.py --npz "$POOL/mean_residual.npz" "$REL/geom_nolen.npz" \
  --out "$OUT_DIR/winner.npz"
python scripts/stack_layers.py --npz "$OUT_DIR/winner.npz" "$OUT_DIR/belief.npz" \
  --out "$OUT_DIR/winner_belief.npz"

python scripts/ridge_screen.py \
  --npz "$OUT_DIR"/*.npz "$POOL/mean.npz" "$POOL/surface_length.npz" \
  --bootstrap 1000 --ref winner --out "$RUN_ROOT/belief_shift.json"
echo "[$(date)] belief done"
