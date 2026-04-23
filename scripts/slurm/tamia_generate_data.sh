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
# Parallel data generation.
#
# Dataset: 172,283 GSM8K problems, ~648K total steps.
# Clean distribution zone: steps 0-375K (~28.9% correct).
# Tail zone (375K+): increasingly correct-only — not used.
#
# Layout:
#   Eval   : offset   0 →  90K  (90K natural steps → ~50K balanced at eval time)
#   Train  : offset  90K → 450K (4 × 90K = 360K steps, all in clean zone)
#
# Eval is carved out FIRST so training never touches those rows.
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="$HOME/projects/aip-azouaq/$USER"
SCRATCH_DATA="$SCRATCH/cot-checker/probe_data"
CKPT="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"

cd "$PROJECT_DIR"
mkdir -p logs "$SCRATCH_DATA"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2

source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# [1/3] Eval shard first — offset 0, 90K natural steps.
# At ~28.9% correct this yields ~26K correct + ~64K incorrect.
# The experiment script balances to 25K + 25K = 50K at eval time.
# ---------------------------------------------------------------------------
echo "=== [1/3] Generating eval shard (GPU 0, offset 0) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/generate_probe_data.py \
    --checkpoint "$CKPT" \
    --output     "$SCRATCH_DATA/eval_held_out.npz" \
    --offset     0 \
    --max-steps  90000 \
    --batch-size 64 \
    --max-seq-len 2048 \
    --device cuda

echo ""

# ---------------------------------------------------------------------------
# [2/3] Training shards — offset 90K onward, 4 × 90K = 360K steps.
# All within the clean distribution zone (0-375K).
# ---------------------------------------------------------------------------
EVAL_SIZE=90000
SHARD_SIZE=90000
N_SHARDS=4

echo "=== [2/3] Generating training shards (4 GPUs in parallel) ==="
PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
    OFFSET=$((EVAL_SIZE + i * SHARD_SIZE))
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
echo "=== [3/3] Merging training shards ==="
python scripts/slurm/merge_shards.py \
    --inputs  $(for i in $(seq 0 $((N_SHARDS-1))); do echo "$SCRATCH_DATA/train_shard_${i}.npz"; done) \
    --output  "$SCRATCH_DATA/train_full.npz"

echo ""
echo "=== Done ==="
ls -lh "$SCRATCH_DATA"

FINAL="$STORE/probe_data"
mkdir -p "$FINAL"
cp "$SCRATCH_DATA/train_full.npz"    "$FINAL/"
cp "$SCRATCH_DATA/eval_held_out.npz" "$FINAL/"
echo "Copied final data to $FINAL"
