#!/bin/bash
#SBATCH --job-name=cot-processbench
#SBATCH --account=aip-azouaq
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=logs/processbench_%j.out
#SBATCH --error=logs/processbench_%j.err

# ---------------------------------------------------------------------------
# ProcessBench evaluation on TWO splits in parallel:
#   GPU 0 — gsm8k  (400 solutions, ~1,500-2,000 steps, ~5 min encode)
#   GPU 1 — math   (1,000 solutions, ~4,000-6,000 steps, ~15 min encode)
#
# Step 1: Both encoding runs in parallel (one per GPU).
# Step 2: Evaluate all trained probe checkpoints on each split and print
#         comparison tables, including ProcessBench published reference numbers.
#
# Prerequisites:
#   - SSAE checkpoint in $STORE/checkpoints/
#   - Trained probe checkpoints in $STORE/results/ (probe_seed{42..45}.pt and
#     linear_probe_seed{42..45}.pt from tamia_train_probe.sh + tamia_baselines.sh)
#   - ProcessBench dataset cached in $STORE/hf_cache
#
# Login-node pre-export (run ONCE before submitting this job):
#   module load StdEnv/2023 gcc arrow/24.0.0 python/3.11
#   source $HOME/venvs/cot/bin/activate
#   export HF_HOME=$SCRATCH/hf_cache
#   python scripts/slurm/export_processbench.py --out-dir $SCRATCH/cot-checker/processbench
#
# Submit: sbatch scripts/slurm/tamia_processbench.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="$HOME/projects/aip-azouaq/$USER"
SCRATCH_PB="$SCRATCH/cot-checker/processbench"

CKPT="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"

LATENTS_GSM="$SCRATCH_PB/processbench_gsm8k.npz"
LATENTS_MATH="$SCRATCH_PB/processbench_math.npz"

cd "$PROJECT_DIR"
mkdir -p logs "$SCRATCH_PB"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2

source "$HOME/venvs/cot/bin/activate"

# Qwen model weights are in $STORE/hf_cache (downloaded by tamia_download_qwen.sh).
# ProcessBench dataset is in $SCRATCH/hf_cache (downloaded on the login node).
# HF_HOME points at $STORE so the SSAE tokenizer is found.
# TRANSFORMERS_OFFLINE=1 keeps Qwen fully offline.
# HF_DATASETS_OFFLINE is intentionally NOT set: with no internet the datasets
# library falls back to the local cache_dir automatically.
export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# Step 1: Encode both splits in parallel, one GPU each.
# ---------------------------------------------------------------------------
echo "=== [1/3] Encoding both ProcessBench splits in parallel ==="
echo "  GPU 0: gsm8k  -> $LATENTS_GSM"
echo "  GPU 1: math   -> $LATENTS_MATH"
echo ""

CUDA_VISIBLE_DEVICES=0 python scripts/encode_processbench.py \
    --checkpoint "$CKPT" \
    --split      gsm8k \
    --data-file  "$SCRATCH_PB/processbench_gsm8k.jsonl" \
    --output     "$LATENTS_GSM" \
    --batch-size 32 \
    --max-seq-len 2048 \
    --device     cuda \
    > "$SCRATCH_PB/encode_gsm8k.log" 2>&1 &
PID_GSM=$!

CUDA_VISIBLE_DEVICES=1 python scripts/encode_processbench.py \
    --checkpoint "$CKPT" \
    --split      math \
    --data-file  "$SCRATCH_PB/processbench_math.jsonl" \
    --output     "$LATENTS_MATH" \
    --batch-size 32 \
    --max-seq-len 2048 \
    --device     cuda \
    > "$SCRATCH_PB/encode_math.log" 2>&1 &
PID_MATH=$!

FAILED=0
if wait "$PID_GSM"; then
    echo "  gsm8k encode finished OK"
    cat "$SCRATCH_PB/encode_gsm8k.log"
else
    echo "  ERROR: gsm8k encode failed -- see $SCRATCH_PB/encode_gsm8k.log"
    cat "$SCRATCH_PB/encode_gsm8k.log"
    FAILED=$((FAILED + 1))
fi

if wait "$PID_MATH"; then
    echo "  math encode finished OK"
    cat "$SCRATCH_PB/encode_math.log"
else
    echo "  ERROR: math encode failed -- see $SCRATCH_PB/encode_math.log"
    cat "$SCRATCH_PB/encode_math.log"
    FAILED=$((FAILED + 1))
fi

[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED encode(s) failed. Aborting."; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# Step 2: Collect probe checkpoints.
# ---------------------------------------------------------------------------
MLP_CKPTS=()
LINEAR_CKPTS=()
for seed in 42 43 44 45; do
    mlp="$STORE/results/probe_seed${seed}.pt"
    lin="$STORE/results/linear_probe_seed${seed}.pt"
    [ -f "$mlp" ] && MLP_CKPTS+=("$mlp")
    [ -f "$lin" ] && LINEAR_CKPTS+=("$lin")
done

if [ ${#MLP_CKPTS[@]} -eq 0 ] && [ ${#LINEAR_CKPTS[@]} -eq 0 ]; then
    echo "ERROR: No probe checkpoints found in $STORE/results/"
    echo "  Run tamia_train_probe.sh and tamia_baselines.sh first."
    exit 1
fi

echo "=== [2/3] Evaluating on ProcessBench GSM8K ==="
echo "  MLP checkpoints   : ${#MLP_CKPTS[@]}"
echo "  Linear checkpoints: ${#LINEAR_CKPTS[@]}"
echo ""

EVAL_ARGS_GSM=(--latents "$LATENTS_GSM" --device cuda)
[ ${#MLP_CKPTS[@]} -gt 0 ]    && EVAL_ARGS_GSM+=(--checkpoints        "${MLP_CKPTS[@]}")
[ ${#LINEAR_CKPTS[@]} -gt 0 ] && EVAL_ARGS_GSM+=(--linear-checkpoints "${LINEAR_CKPTS[@]}")

CUDA_VISIBLE_DEVICES=0 python scripts/eval_processbench.py "${EVAL_ARGS_GSM[@]}"

echo ""
echo "=== [3/3] Evaluating on ProcessBench MATH ==="
echo ""

EVAL_ARGS_MATH=(--latents "$LATENTS_MATH" --device cuda)
[ ${#MLP_CKPTS[@]} -gt 0 ]    && EVAL_ARGS_MATH+=(--checkpoints        "${MLP_CKPTS[@]}")
[ ${#LINEAR_CKPTS[@]} -gt 0 ] && EVAL_ARGS_MATH+=(--linear-checkpoints "${LINEAR_CKPTS[@]}")

CUDA_VISIBLE_DEVICES=0 python scripts/eval_processbench.py "${EVAL_ARGS_MATH[@]}"

# ---------------------------------------------------------------------------
# Copy latents and logs to project space.
# ---------------------------------------------------------------------------
FINAL="$STORE/probe_data"
mkdir -p "$FINAL"
cp "$LATENTS_GSM"  "$FINAL/"
cp "$LATENTS_MATH" "$FINAL/"
cp "$SCRATCH_PB"/encode_*.log "$FINAL/" 2>/dev/null || true

echo ""
echo "Latents saved -> $FINAL"
echo "=== Done ==="
