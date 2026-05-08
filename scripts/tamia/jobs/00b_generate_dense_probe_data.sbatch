#!/bin/bash
#SBATCH --job-name=ptb-00b-dense
#SBATCH --account=aip-azouaq
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 00b: Generate dense probe data (one-time prerequisite)
#
# Creates dense_eval_held_out.npz and dense_train_full.npz -- the raw backbone
# hidden states h_k for each Math-Shepherd step, with correctness labels.
# These are the input to Stage 03 (extract PTB latents).
#
# dense_eval_held_out : first 90K MS steps (offset=0)
# dense_train_full    : next 360K MS steps (offset=90000)
#
# Requires the HF dataset cache (peiyi9979/Math-Shepherd) at $HF_HOME.
# Since eval_held_out.npz (SSAE latents) already exists, the cache is present.
#
# Expected runtime: ~3-5h for 450K steps on one H100.
#
# Submit once, before Stage 00:
#   sbatch scripts/tamia/jobs/00b_generate_dense_probe_data.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
CKPT_SSAE="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"
PROBE_DATA="$STORE/probe_data"

cd "$PROJECT_DIR"
mkdir -p results/logs "$PROBE_DATA"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "======================================================="
echo "  Stage 00b: Generate dense probe data"
echo "======================================================="
echo ""

# ---------------------------------------------------------------------------
# Eval split: first 90K steps (offset=0)
# Matches the offset used in add_ms_solution_ids.py for eval
# ---------------------------------------------------------------------------
EVAL_OUT="$PROBE_DATA/dense_eval_held_out.npz"
if [ -f "$EVAL_OUT" ]; then
    echo "  dense_eval_held_out.npz already exists, skipping."
else
    echo "  Generating dense_eval_held_out.npz (90K steps, offset=0) ..."
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_probe_data.py \
        --checkpoint  "$CKPT_SSAE" \
        --output      "$EVAL_OUT" \
        --max-steps   90000 \
        --offset      0 \
        --batch-size  16 \
        --max-seq-len 256 \
        --encoding    dense \
        --device      cuda
    echo "  Done: $EVAL_OUT"
fi
echo ""

# ---------------------------------------------------------------------------
# Train split: next 360K steps (offset=90000)
# Matches the offset used in add_ms_solution_ids.py for train
# ---------------------------------------------------------------------------
TRAIN_OUT="$PROBE_DATA/dense_train_full.npz"
if [ -f "$TRAIN_OUT" ]; then
    echo "  dense_train_full.npz already exists, skipping."
else
    echo "  Generating dense_train_full.npz (360K steps, offset=90000) ..."
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_probe_data.py \
        --checkpoint  "$CKPT_SSAE" \
        --output      "$TRAIN_OUT" \
        --max-steps   360000 \
        --offset      90000 \
        --batch-size  16 \
        --max-seq-len 256 \
        --encoding    dense \
        --device      cuda
    echo "  Done: $TRAIN_OUT"
fi
echo ""

echo "======================================================="
echo "  Stage 00b complete."
python3 -c "
import numpy as np
for path in ['$EVAL_OUT', '$TRAIN_OUT']:
    d = np.load(path)
    print(f'  {path.split(\"/\")[-1]}  shape={d[\"latents\"].shape}  keys={list(d.files)}')
"
echo ""
echo "  Next: sbatch scripts/tamia/jobs/00_ptb_sanity.sbatch"
echo "======================================================="
