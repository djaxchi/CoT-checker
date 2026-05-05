#!/bin/bash
#SBATCH --job-name=cot-ssae-c2
#SBATCH --account=aip-azouaq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=h100:4
#SBATCH --output=logs/ssae_c2_%j.out
#SBATCH --error=logs/ssae_c2_%j.err

# ---------------------------------------------------------------------------
# End-to-end SSAE c=2 pipeline + 5-encoding mechanistic comparison.
#
# Architecture: sparsity_factor=2 (1792-dim latent), ReLU + L1 sparsity.
# No TopK: the soft L1 penalty lets each step activate as many features
# as it actually needs — the right inductive bias for monosemanticity.
# Frozen encoder: halves training time vs. full backprop through the
# backbone; the monosemanticity test is in the SSAE latent space, not
# whether Qwen's weights fine-tune.
#
# GPU usage:
#   Training   [1]: GPU 0 only (~7-9h)  — single-GPU, can't parallelize
#   Encoding   [2]: GPU 0               — eval shard, ~10min
#   Encoding   [3]: GPUs 0-3 parallel   — 4 train shards, ~10min
#   Probes     [5]: GPUs 0-3 parallel   — 4 linear + 4 MLP, ~15min
#   PB encode  [6]: GPU 0               — ProcessBench, ~5min
#   Viz        [7]: CPU                 — comparison figures, ~5min
#
# Steps:
#   [1/7] Train SSAE c=2 (GPU 0, frozen encoder, ReLU+L1, bfloat16, 8 epochs)
#   [2/7] Encode eval shard       (GPU 0, 90K steps)
#   [3/7] Encode train shards     (4 GPUs parallel, 4 × 90K steps)
#   [4/7] Merge train shards      (CPU)
#   [5/7] Train probes            (4 GPUs parallel, 4 linear + 4 MLP seeds)
#   [6/7] Encode ProcessBench     (GPU 0)
#   [7/7] Regenerate comparison   (CPU, 5-encoding PCA/histogram figures)
#
# Prerequisites: same as tamia_train_ssae_c4.sh — needs gsm8k_385K_train.json
# and the previous comparison outputs (dense, c=1, c=4, actsae npz/probes).
#
# Submit: sbatch scripts/slurm/tamia_ssae_c2_full.sh
# Fetch:  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_comparison/ ./results/mechanistic_comparison/
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

SCRATCH_DATA="$SCRATCH_BASE/probe_data_c2"
SCRATCH_RESULTS="$SCRATCH_BASE/results_c2"
NEW_CKPT_DIR="$SCRATCH_BASE/ssae_c2"
CKPT_DIR="$STORE/checkpoints"

TRAIN_DATA="$STORE/data/gsm8k_385K_train.json"
VAL_DATA="$STORE/data/gsm8k_385K_valid.json"
NEW_CKPT="$NEW_CKPT_DIR/best.pt"

C2_EVAL="$SCRATCH_DATA/c2_eval_held_out.npz"
C2_TRAIN="$SCRATCH_DATA/c2_train_full.npz"

PB_JSONL="$SCRATCH_BASE/processbench/processbench_gsm8k.jsonl"
PB_C2="$SCRATCH_BASE/processbench/processbench_c2_gsm8k.npz"
VIZ_OUT="$PROJECT_DIR/results/mechanistic_comparison"

cd "$PROJECT_DIR"
mkdir -p logs "$SCRATCH_DATA" "$SCRATCH_RESULTS" "$NEW_CKPT_DIR" "$VIZ_OUT"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2

source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "=== Pre-flight checks ==="
[ ! -f "$TRAIN_DATA" ] && { echo "ERROR: $TRAIN_DATA not found."; exit 1; }
[ ! -f "$PB_JSONL"   ] && { echo "ERROR: $PB_JSONL not found. Run export_processbench.py first."; exit 1; }

VAL_FLAG=""
[ -f "$VAL_DATA" ] && VAL_FLAG="--val-data $VAL_DATA" \
    || echo "  WARNING: validation data not found, will auto-split 5% of train."

echo "  Training data : $TRAIN_DATA"
echo "  Val data      : ${VAL_DATA:-auto-split}"
echo ""

# ---------------------------------------------------------------------------
# [1/7] Train SSAE c=2 — ReLU+L1, frozen encoder, bfloat16
# ---------------------------------------------------------------------------
echo "=== [1/7] Training SSAE c=2 ReLU+L1 (GPU 0, frozen encoder) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/train_ssae.py \
    --data            "$TRAIN_DATA" \
    ${VAL_FLAG:+--val-data "$VAL_DATA"} \
    --output-dir      "$NEW_CKPT_DIR" \
    --model-id        Qwen/Qwen2.5-0.5B \
    --sparsity-factor 2 \
    --freeze-encoder \
    --dtype           bfloat16 \
    --epochs          8 \
    --batch-size      16 \
    --grad-accum      8 \
    --lr              1e-4 \
    --min-lr          1e-5 \
    --warmup-steps    500 \
    --num-workers     4 \
    --device          cuda

echo "  Checkpoint: $NEW_CKPT"
echo ""

# ---------------------------------------------------------------------------
# [2/7] Encode eval shard (offset 0, 90K steps, GPU 0)
# ---------------------------------------------------------------------------
echo "=== [2/7] Encoding eval shard (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/generate_probe_data.py \
    --checkpoint  "$NEW_CKPT" \
    --output      "$C2_EVAL" \
    --offset      0 \
    --max-steps   90000 \
    --batch-size  32 \
    --max-seq-len 2048 \
    --encoding    sparse \
    --device      cuda
echo ""

# ---------------------------------------------------------------------------
# [3/7] Encode train shards in parallel (4 GPUs, 4 × 90K)
# ---------------------------------------------------------------------------
echo "=== [3/7] Encoding train shards (4 GPUs in parallel) ==="
PIDS=()
for i in 0 1 2 3; do
    OFFSET=$(( 90000 + i * 90000 ))
    OUT="$SCRATCH_DATA/c2_train_shard_${i}.npz"
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_probe_data.py \
        --checkpoint  "$NEW_CKPT" \
        --output      "$OUT" \
        --offset      "$OFFSET" \
        --max-steps   90000 \
        --batch-size  32 \
        --max-seq-len 2048 \
        --encoding    sparse \
        --device      cuda \
        > "$SCRATCH_DATA/c2_shard_${i}.log" 2>&1 &
    PIDS+=($!)
done
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" && echo "  Shard $i OK" \
        || { echo "  ERROR: shard $i failed (see c2_shard_${i}.log)"; FAILED=$((FAILED+1)); }
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED shard(s) failed."; exit 1; }
echo ""

# ---------------------------------------------------------------------------
# [4/7] Merge train shards
# ---------------------------------------------------------------------------
echo "=== [4/7] Merging train shards ==="
python scripts/slurm/merge_shards.py \
    --inputs "$SCRATCH_DATA/c2_train_shard_0.npz" \
             "$SCRATCH_DATA/c2_train_shard_1.npz" \
             "$SCRATCH_DATA/c2_train_shard_2.npz" \
             "$SCRATCH_DATA/c2_train_shard_3.npz" \
    --output "$C2_TRAIN"
echo ""

# ---------------------------------------------------------------------------
# [5/7] Train probes — linear + MLP, 4 seeds each, all 4 GPUs in parallel
# ---------------------------------------------------------------------------
echo "=== [5/7] Training probes (8 jobs across 4 GPUs) ==="
SEEDS=(42 43 44 45)
PIDS=()

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    CUDA_VISIBLE_DEVICES=$i python scripts/experiment_linear_probe.py \
        --train-data "$C2_TRAIN" --eval-data "$C2_EVAL" \
        --output     "$SCRATCH_RESULTS/c2_linear_probe_seed${SEED}.pt" \
        --seed "$SEED" --epochs 50 --batch-size 512 --device cuda \
        > "$SCRATCH_RESULTS/c2_linear_probe_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done
# MLP probes on the same GPUs — start immediately, linear probes are fast
# and will free GPU memory before MLP probes need it
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    CUDA_VISIBLE_DEVICES=$i python scripts/experiment_full_clean.py \
        --train-data "$C2_TRAIN" --eval-data "$C2_EVAL" \
        --output     "$SCRATCH_RESULTS/c2_probe_seed${SEED}.pt" \
        --seed "$SEED" --epochs 50 --batch-size 512 --device cuda \
        > "$SCRATCH_RESULTS/c2_probe_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED+1))
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED probe job(s) failed."; exit 1; }

# Print quick accuracy summary
echo "  Probe results:"
for SEED in "${SEEDS[@]}"; do
    grep "SUMMARY" "$SCRATCH_RESULTS/c2_linear_probe_seed${SEED}.log" 2>/dev/null \
        | sed "s/^/    linear seed=${SEED}: /" || true
    grep "SUMMARY" "$SCRATCH_RESULTS/c2_probe_seed${SEED}.log" 2>/dev/null \
        | sed "s/^/    MLP    seed=${SEED}: /" || true
done
echo ""

# ---------------------------------------------------------------------------
# [6/7] Encode ProcessBench with SSAE c=2 (GPU 0)
# ---------------------------------------------------------------------------
echo "=== [6/7] Encoding ProcessBench with SSAE c=2 ==="
CUDA_VISIBLE_DEVICES=0 python scripts/encode_processbench.py \
    --checkpoint "$NEW_CKPT" \
    --data-file  "$PB_JSONL" \
    --output     "$PB_C2" \
    --encoding   ssae \
    --batch-size 32 \
    --max-seq-len 2048 \
    --device     cuda
echo ""

# ---------------------------------------------------------------------------
# [7/7] Regenerate 5-encoding comparison figures
# ---------------------------------------------------------------------------
echo "=== [7/7] Regenerating mechanistic comparison figures ==="

# Save checkpoint and probes to persistent storage
mkdir -p "$CKPT_DIR" "$STORE/results_c2"
cp "$NEW_CKPT" "$CKPT_DIR/ssae_c2_relu_l1.pt" 2>/dev/null || true
cp "$SCRATCH_RESULTS"/c2_linear_probe_seed*.{pt,log} "$STORE/results_c2/" 2>/dev/null || true
cp "$SCRATCH_RESULTS"/c2_probe_seed*.{pt,log}        "$STORE/results_c2/" 2>/dev/null || true

# Build probe lists
e1_probes=$(printf '"%s",' $(for s in 42 43 44 45; do echo "$STORE/results/dense_linear_probe_seed${s}.pt"; done) | sed 's/,$//')
e2_probes=$(printf '"%s",' $(for s in 42 43 44 45; do echo "$STORE/results/linear_probe_seed${s}.pt"; done) | sed 's/,$//')
e3_probes=$(printf '"%s",' $(for s in 42 43 44 45; do echo "$STORE/results_c2/c2_linear_probe_seed${s}.pt"; done) | sed 's/,$//')
e4_probes=$(printf '"%s",' $(for s in 42 43 44 45; do
    f="$STORE/results_c4/c4_linear_probe_seed${s}.pt"
    [ -f "$f" ] || f="$SCRATCH_BASE/results_c4/c4_linear_probe_seed${s}.pt"
    echo "$f"
done) | sed 's/,$//')
e5_probes=$(printf '"%s",' $(for s in 42 43 44 45; do echo "$SCRATCH_BASE/results_actsae/actsae_linear_probe_seed${s}.pt"; done) | sed 's/,$//')

cat > "$VIZ_OUT/config.json" <<JSON
{
  "n_samples_ms": 5000,
  "n_samples_pb": 0,
  "encodings": [
    {
      "label":        "Dense h_k",
      "ms_npz":       "${SCRATCH_BASE}/probe_data/dense_eval_held_out.npz",
      "pb_npz":       "${SCRATCH_BASE}/processbench/processbench_dense_gsm8k.npz",
      "probes":       [${e1_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "SSAE c=1",
      "ms_npz":       "${SCRATCH_BASE}/probe_data/eval_held_out.npz",
      "pb_npz":       "${SCRATCH_BASE}/processbench/processbench_gsm8k.npz",
      "probes":       [${e2_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "SSAE c=2 ReLU+L1",
      "ms_npz":       "${C2_EVAL}",
      "pb_npz":       "${PB_C2}",
      "probes":       [${e3_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "SSAE c=4 TopK*",
      "ms_npz":       "${SCRATCH_BASE}/probe_data_c4/c4_eval_held_out.npz",
      "pb_npz":       "${SCRATCH_BASE}/processbench/processbench_c4_gsm8k.npz",
      "probes":       [${e4_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "Activation SAE",
      "ms_npz":       "${SCRATCH_BASE}/results_actsae/actsae_eval_held_out.npz",
      "pb_npz":       "${SCRATCH_BASE}/results_actsae/processbench_actsae_gsm8k.npz",
      "probes":       [${e5_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    }
  ]
}
JSON

python scripts/compare_mechanistic_viz.py \
    --config     "$VIZ_OUT/config.json" \
    --output-dir "$VIZ_OUT" \
    --reducer    pca \
    --dpi        150

echo ""
echo "=== All done ==="
echo ""
echo "  Checkpoint : $CKPT_DIR/ssae_c2_relu_l1.pt"
echo "  Figures    : $VIZ_OUT/"
echo ""
echo "  Fetch figures with:"
echo "  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_comparison/ ./results/mechanistic_comparison/"
