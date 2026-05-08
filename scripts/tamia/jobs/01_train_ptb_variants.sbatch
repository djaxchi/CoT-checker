#!/bin/bash
#SBATCH --job-name=ptb-01-train
#SBATCH --account=aip-azouaq
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 01: Cache transitions + train 4 PTB variants
#
# Uses all 4 H100 GPUs:
#   GPU 0+1+2+3 (parallel shards): cache transition hidden states
#   GPU 0: ptb_no_l1         -- transition-only baseline (no sparsity penalty)
#   GPU 1: ptb_dwa_calibrated -- DWA with empirically calibrated L1 target
#   GPU 2: ptb_active_fraction -- DWA on active-dim fraction
#   GPU 3: ptb_topk           -- exact k-sparse, no L1
#
# Artifacts per variant:
#   $PTB_CKPT_BASE/<name>/best.pt
#   $PTB_CKPT_BASE/<name>/calibration.json
#   $PTB_CKPT_BASE/<name>/validity_report.json
#   $PTB_CKPT_BASE/<name>/train_metrics.csv
#   $PTB_CKPT_BASE/<name>/train.log  (stdout)
#
# Decision (Stage 02): read validity reports before extracting latents.
#
# Submit:
#   sbatch scripts/tamia/jobs/01_train_ptb_variants.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

DATA_TRAIN="$STORE/data/gsm8k_385K_train.json"
DATA_VALID="$STORE/data/gsm8k_385K_valid.json"
CKPT_SSAE="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"

TRANS_DIR="$SCRATCH_BASE/probe_data/transitions"
PTB_CKPT_BASE="$STORE/results/checkpoints/ptb_robust"

cd "$PROJECT_DIR"
mkdir -p results/logs "$TRANS_DIR" "$PTB_CKPT_BASE"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------------------
# Cache training transitions (4 parallel GPU shards, positive-only)
# ---------------------------------------------------------------------------
echo "=== [1/3] Cache training transitions ==="
TRANS_TRAIN="$TRANS_DIR/transition_train_positive.npz"

if [ -f "$TRANS_TRAIN" ]; then
    echo "  Already exists, skipping."
else
    SHARD_PIDS=()
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python scripts/cache_transition_hidden_states.py \
            --checkpoint   "$CKPT_SSAE" \
            --data-file    "$DATA_TRAIN" \
            --output       "$TRANS_DIR/trans_train_shard${i}.npz" \
            --positive-only \
            --shard-idx    $i \
            --n-shards     4 \
            --batch-size   32 \
            --max-seq-len  256 \
            --device       cuda \
            > "$TRANS_DIR/shard${i}_train.log" 2>&1 &
        SHARD_PIDS+=($!)
    done
    FAILED=0
    for i in "${!SHARD_PIDS[@]}"; do
        wait "${SHARD_PIDS[$i]}" || FAILED=$((FAILED+1))
    done
    [ "$FAILED" -gt 0 ] && { echo "ERROR: training shard(s) failed"; exit 1; }
    python scripts/merge_transition_shards.py \
        --inputs "$TRANS_DIR/trans_train_shard0.npz" "$TRANS_DIR/trans_train_shard1.npz" \
                 "$TRANS_DIR/trans_train_shard2.npz" "$TRANS_DIR/trans_train_shard3.npz" \
        --output "$TRANS_TRAIN"
    rm -f "$TRANS_DIR/trans_train_shard"*.npz
fi
echo ""

# ---------------------------------------------------------------------------
# Cache validation transitions
# ---------------------------------------------------------------------------
echo "=== [2/3] Cache validation transitions ==="
TRANS_VAL="$TRANS_DIR/transition_val.npz"

if [ -f "$TRANS_VAL" ]; then
    echo "  Already exists, skipping."
else
    SHARD_PIDS=()
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python scripts/cache_transition_hidden_states.py \
            --checkpoint  "$CKPT_SSAE" \
            --data-file   "$DATA_VALID" \
            --output      "$TRANS_DIR/trans_val_shard${i}.npz" \
            --shard-idx   $i \
            --n-shards    4 \
            --batch-size  32 \
            --max-seq-len 256 \
            --device      cuda \
            > "$TRANS_DIR/shard${i}_val.log" 2>&1 &
        SHARD_PIDS+=($!)
    done
    FAILED=0
    for i in "${!SHARD_PIDS[@]}"; do
        wait "${SHARD_PIDS[$i]}" || FAILED=$((FAILED+1))
    done
    [ "$FAILED" -gt 0 ] && { echo "ERROR: val shard(s) failed"; exit 1; }
    python scripts/merge_transition_shards.py \
        --inputs "$TRANS_DIR/trans_val_shard0.npz" "$TRANS_DIR/trans_val_shard1.npz" \
                 "$TRANS_DIR/trans_val_shard2.npz" "$TRANS_DIR/trans_val_shard3.npz" \
        --output "$TRANS_VAL"
    rm -f "$TRANS_DIR/trans_val_shard"*.npz
fi
echo ""

# ---------------------------------------------------------------------------
# Train 4 PTB variants in parallel (one per GPU)
# ---------------------------------------------------------------------------
echo "=== [3/3] Train PTB variants ==="

_TRAIN_ARGS="--train-data $TRANS_TRAIN --val-data $TRANS_VAL --d 896 --epochs 30 --batch-size 2048 --lr 1e-3 --device cuda --calib-batches 50"

PIDS=()

mkdir -p "$PTB_CKPT_BASE/no_l1"
if [ -f "$PTB_CKPT_BASE/no_l1/best.pt" ]; then
    echo "  [no_l1] Checkpoint exists, skipping."
else
    echo "  [no_l1] Training on GPU 0 ..."
    CUDA_VISIBLE_DEVICES=0 python scripts/train_predictive_transition_bottleneck.py \
        $_TRAIN_ARGS --output-dir "$PTB_CKPT_BASE/no_l1" \
        --dwa-mode no_l1 \
        > "$PTB_CKPT_BASE/no_l1/train.log" 2>&1 &
    PIDS+=($!)
fi

mkdir -p "$PTB_CKPT_BASE/dwa_calibrated"
if [ -f "$PTB_CKPT_BASE/dwa_calibrated/best.pt" ]; then
    echo "  [dwa_calibrated] Checkpoint exists, skipping."
else
    echo "  [dwa_calibrated] Training on GPU 1 ..."
    CUDA_VISIBLE_DEVICES=1 python scripts/train_predictive_transition_bottleneck.py \
        $_TRAIN_ARGS --output-dir "$PTB_CKPT_BASE/dwa_calibrated" \
        --dwa-mode dwa_calibrated_mean_l1 \
        > "$PTB_CKPT_BASE/dwa_calibrated/train.log" 2>&1 &
    PIDS+=($!)
fi

mkdir -p "$PTB_CKPT_BASE/active_fraction"
if [ -f "$PTB_CKPT_BASE/active_fraction/best.pt" ]; then
    echo "  [active_fraction] Checkpoint exists, skipping."
else
    echo "  [active_fraction] Training on GPU 2 ..."
    CUDA_VISIBLE_DEVICES=2 python scripts/train_predictive_transition_bottleneck.py \
        $_TRAIN_ARGS --output-dir "$PTB_CKPT_BASE/active_fraction" \
        --dwa-mode dwa_active_fraction --l1-target-active-frac 0.25 \
        > "$PTB_CKPT_BASE/active_fraction/train.log" 2>&1 &
    PIDS+=($!)
fi

mkdir -p "$PTB_CKPT_BASE/topk"
if [ -f "$PTB_CKPT_BASE/topk/best.pt" ]; then
    echo "  [topk] Checkpoint exists, skipping."
else
    echo "  [topk] Training on GPU 3 ..."
    CUDA_VISIBLE_DEVICES=3 python scripts/train_predictive_transition_bottleneck.py \
        $_TRAIN_ARGS --output-dir "$PTB_CKPT_BASE/topk" \
        --dwa-mode topk --l1-k 50 \
        > "$PTB_CKPT_BASE/topk/train.log" 2>&1 &
    PIDS+=($!)
fi

for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "WARNING: variant pid=$pid exited non-zero"
done
echo "  All variants done."
echo ""

# ---------------------------------------------------------------------------
# Print compact validity summary
# ---------------------------------------------------------------------------
echo "=== Validity flags (quick view) ==="
for name in no_l1 dwa_calibrated active_fraction topk; do
    vr="$PTB_CKPT_BASE/$name/validity_report.json"
    if [ -f "$vr" ]; then
        echo "  [$name] $(python3 -c "import json; d=json.load(open('$vr')); print('flags:', d['flags'][:2] if len(d['flags'])>2 else d['flags'])")"
    else
        echo "  [$name] no validity_report.json"
    fi
done
echo ""
echo "======================================================="
echo "  Stage 01 complete."
echo "  Checkpoints: $PTB_CKPT_BASE/{no_l1,dwa_calibrated,active_fraction,topk}/"
echo "  Next: sbatch scripts/tamia/jobs/02_collect_ptb_validity.sbatch"
echo "======================================================="
