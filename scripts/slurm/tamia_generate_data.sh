#!/bin/bash
#SBATCH --job-name=cot-generate
#SBATCH --account=aip-azouaq
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # 4 shards × 4 CPU workers each
#SBATCH --mem=128G
#SBATCH --gpus=h100:4               # must use all 4 GPUs on an H100 node
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

# ---------------------------------------------------------------------------
# Parallel data generation: splits Math-Shepherd into 4 shards,
# runs one Python process per GPU simultaneously, then merges.
#
# TamIA compute nodes have no internet — run tamia_setup.sh first.
#
# Submit: sbatch scripts/slurm/tamia_generate_data.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="$HOME/projects/aip-azouaq/$USER"
SCRATCH_DATA="$SCRATCH/cot-checker/probe_data"
CKPT="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"

cd "$PROJECT_DIR"
mkdir -p logs "$SCRATCH_DATA"

module purge
module load StdEnv/2023 python/3.11 cuda/12.2

source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export HF_DATASETS_OFFLINE=1   # dataset already cached — no network needed
export TRANSFORMERS_OFFLINE=1  # model weights already cached — no network needed

# ---------------------------------------------------------------------------
# Training shards: 4 × 125K steps = 500K total, each on a dedicated GPU.
# Verified dataset size: 172,283 GSM8K problems, ~648K total steps.
# Steps 0-499K → training. Steps 500K onward → untouched held-out pool.
# ---------------------------------------------------------------------------
SHARD_SIZE=125000
N_SHARDS=4

echo "=== [1/3] Generating training shards (4 GPUs in parallel) ==="
PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
    OFFSET=$((i * SHARD_SIZE))
    OUT="$SCRATCH_DATA/train_shard_${i}.npz"
    echo "  GPU $i: offset=$OFFSET, steps=$SHARD_SIZE → $OUT"
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_probe_data.py \
        --checkpoint "$CKPT" \
        --output     "$OUT" \
        --offset     "$OFFSET" \
        --max-steps  "$SHARD_SIZE" \
        --batch-size 64 \
        --max-seq-len 2048 \
        --device cuda \
        > "$SCRATCH_DATA/shard_${i}.log" 2>&1 &
    PIDS+=($!)
done

# Wait for all shards and check exit codes
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  Shard $i finished OK"
    else
        echo "  ERROR: shard $i failed — check $SCRATCH_DATA/shard_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED shard(s) failed. Aborting."
    exit 1
fi

echo ""
echo "=== [2/3] Generating eval shard (GPU 0, held-out) ==="
# Offset = 100K (immediately past training data). ~548K steps remain;
# we take 20K for a robust held-out set, leaving ~528K never touched.
EVAL_OFFSET=$((N_SHARDS * SHARD_SIZE))   # 500000
# Collect all remaining steps (~148K) without any subsampling.
# Balancing to 50/50 is done at eval time inside experiment_full_clean.py.
# Set max-steps high enough that the dataset exhausts naturally.
CUDA_VISIBLE_DEVICES=0 python scripts/generate_probe_data.py \
    --checkpoint "$CKPT" \
    --output     "$SCRATCH_DATA/eval_held_out.npz" \
    --offset     "$EVAL_OFFSET" \
    --max-steps  200000 \
    --batch-size 64 \
    --max-seq-len 2048 \
    --device cuda

echo ""
echo "=== [3/3] Merging training shards ==="
python scripts/slurm/merge_shards.py \
    --inputs  $(for i in $(seq 0 $((N_SHARDS-1))); do echo "$SCRATCH_DATA/train_shard_${i}.npz"; done) \
    --output  "$SCRATCH_DATA/train_full.npz"

echo ""
echo "=== Done ==="
ls -lh "$SCRATCH_DATA"

# Copy final merged files to project space for safekeeping
FINAL="$STORE/probe_data"
mkdir -p "$FINAL"
cp "$SCRATCH_DATA/train_full.npz"    "$FINAL/"
cp "$SCRATCH_DATA/eval_held_out.npz" "$FINAL/"
echo "Copied final data to $FINAL"
