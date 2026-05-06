#!/bin/bash
#SBATCH --job-name=cot-transition
#SBATCH --account=aip-azouaq
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=h100:4
#SBATCH --output=logs/transition_%j.out
#SBATCH --error=logs/transition_%j.err

# ---------------------------------------------------------------------------
# Transition probe analysis across all 5 encodings.
#
# Hypothesis: Δz = z_k - z_{k-1} (feature change between consecutive steps)
# may be more linearly separable for detecting incorrect steps than z_k alone.
#
# Pipeline:
#   [1/3] Re-encode Math-Shepherd eval shard with solution metadata
#         (4 GPUs in parallel: Dense, SSAE c=1, SSAE c=2, SSAE c=4)
#   [2/3] Apply activation SAE to new dense encoding (CPU)
#   [3/3] Run transition probe analysis on both MS + ProcessBench (CPU)
#
# ProcessBench npz files already have solution_ids + step_positions, so
# transition probes can run on them directly without re-encoding.
#
# Prerequisites:
#   - All 5 encoding checkpoints present
#   - ProcessBench npz files in $SCRATCH/cot-checker/processbench/
#   - Existing mechanistic_comparison/config.json
#   - Activation SAE checkpoint at $SCRATCH/cot-checker/results_actsae/actsae.pt
#
# Submit: sbatch scripts/slurm/tamia_transition_probes.sh
# Fetch:  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_comparison/ ./results/mechanistic_comparison/
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

# Checkpoints
CKPT_C1="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"
CKPT_C2="$STORE/checkpoints/ssae_c2_relu_l1.pt"
CKPT_C4="$STORE/checkpoints/ssae_c4_topk40_frozen.pt"
CKPT_ACTSAE="$SCRATCH_BASE/results_actsae/actsae.pt"

# Re-encoded eval shards with solution metadata
META_DIR="$SCRATCH_BASE/probe_data_meta"
DENSE_EVAL_META="$META_DIR/dense_eval_meta.npz"
C1_EVAL_META="$META_DIR/c1_eval_meta.npz"
C2_EVAL_META="$META_DIR/c2_eval_meta.npz"
C4_EVAL_META="$META_DIR/c4_eval_meta.npz"
ACTSAE_EVAL_META="$META_DIR/actsae_eval_meta.npz"

# ProcessBench (already have solution_ids + step_positions)
PB_DENSE="$SCRATCH_BASE/processbench/processbench_dense_gsm8k.npz"
PB_C1="$SCRATCH_BASE/processbench/processbench_gsm8k.npz"
PB_C2="$SCRATCH_BASE/processbench/processbench_c2_gsm8k.npz"
PB_C4="$SCRATCH_BASE/processbench/processbench_c4_gsm8k.npz"
PB_ACTSAE="$SCRATCH_BASE/results_actsae/processbench_actsae_gsm8k.npz"

# Per-step probe checkpoints (from original experiments)
RESULTS="$STORE/results"
RESULTS_C2="$STORE/results_c2"
RESULTS_C4="$STORE/results_c4"
RESULTS_ACTSAE="$SCRATCH_BASE/results_actsae"

# Training data (only needed for dataset streaming to re-encode)
TRAIN_DATA="$STORE/data/gsm8k_385K_train.json"

VIZ_OUT="$PROJECT_DIR/results/mechanistic_comparison"
CONFIG_JSON="$VIZ_OUT/config.json"

cd "$PROJECT_DIR"
mkdir -p logs "$META_DIR" "$VIZ_OUT"

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
for f in "$CKPT_C1" "$CKPT_C2" "$CKPT_C4" "$CKPT_ACTSAE" \
          "$PB_C1" "$PB_C2" "$PB_C4" "$CONFIG_JSON"; do
    [ ! -f "$f" ] && { echo "ERROR: $f not found"; exit 1; }
done

# Dense PB might have been created after the original comparison
if [ ! -f "$PB_DENSE" ]; then
    echo "  WARNING: $PB_DENSE not found -- transition probes on ProcessBench Dense will be skipped"
fi
echo "  Pre-flight OK"
echo ""

# ---------------------------------------------------------------------------
# [1/3] Re-encode Math-Shepherd eval shard with solution metadata (4 GPUs)
# ---------------------------------------------------------------------------
echo "=== [1/3] Re-encoding Math-Shepherd eval shard with solution metadata ==="
echo "  (generate_probe_data.py now saves solution_ids + step_positions)"
PIDS=()

# GPU 0: Dense h_k
CUDA_VISIBLE_DEVICES=0 python scripts/generate_probe_data.py \
    --checkpoint  "$CKPT_C1" \
    --output      "$DENSE_EVAL_META" \
    --offset      0 \
    --max-steps   90000 \
    --batch-size  32 \
    --max-seq-len 2048 \
    --encoding    dense \
    --device      cuda \
    > "$META_DIR/encode_dense_meta.log" 2>&1 &
PIDS+=($!)

# GPU 1: SSAE c=1
CUDA_VISIBLE_DEVICES=1 python scripts/generate_probe_data.py \
    --checkpoint  "$CKPT_C1" \
    --output      "$C1_EVAL_META" \
    --offset      0 \
    --max-steps   90000 \
    --batch-size  32 \
    --max-seq-len 2048 \
    --encoding    sparse \
    --device      cuda \
    > "$META_DIR/encode_c1_meta.log" 2>&1 &
PIDS+=($!)

# GPU 2: SSAE c=2
CUDA_VISIBLE_DEVICES=2 python scripts/generate_probe_data.py \
    --checkpoint  "$CKPT_C2" \
    --output      "$C2_EVAL_META" \
    --offset      0 \
    --max-steps   90000 \
    --batch-size  32 \
    --max-seq-len 2048 \
    --encoding    sparse \
    --device      cuda \
    > "$META_DIR/encode_c2_meta.log" 2>&1 &
PIDS+=($!)

# GPU 3: SSAE c=4
CUDA_VISIBLE_DEVICES=3 python scripts/generate_probe_data.py \
    --checkpoint  "$CKPT_C4" \
    --output      "$C4_EVAL_META" \
    --offset      0 \
    --max-steps   90000 \
    --batch-size  32 \
    --max-seq-len 2048 \
    --encoding    sparse \
    --device      cuda \
    > "$META_DIR/encode_c4_meta.log" 2>&1 &
PIDS+=($!)

FAILED=0
for i in "${!PIDS[@]}"; do
    tags=("dense" "c1" "c2" "c4")
    wait "${PIDS[$i]}" && echo "  ${tags[$i]} OK" \
        || { echo "  ERROR: ${tags[$i]} failed"; FAILED=$((FAILED+1)); }
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED encode job(s) failed"; exit 1; }
echo ""

# ---------------------------------------------------------------------------
# [2/3] Apply activation SAE to new dense encoding (CPU)
# ---------------------------------------------------------------------------
echo "=== [2/3] Applying activation SAE to dense eval (CPU) ==="
python scripts/train_activation_sae.py \
    --encode-only \
    --actsae-checkpoint "$CKPT_ACTSAE" \
    --encode-extra      "${DENSE_EVAL_META}:${ACTSAE_EVAL_META}" \
    --device            cpu
echo ""

# ---------------------------------------------------------------------------
# [3/3] Build per-dataset config and run transition probe analysis (CPU)
# ---------------------------------------------------------------------------
echo "=== [3/3] Running transition probe analysis ==="

# Build probe lists (reuse existing per-step probe checkpoints for comparison)
dense_probes=$(printf '"%s",' \
    "$RESULTS/dense_linear_probe_seed42.pt" \
    "$RESULTS/dense_linear_probe_seed43.pt" \
    "$RESULTS/dense_linear_probe_seed44.pt" \
    "$RESULTS/dense_linear_probe_seed45.pt" | sed 's/,$//')

c1_probes=$(printf '"%s",' \
    "$RESULTS/linear_probe_seed42.pt" \
    "$RESULTS/linear_probe_seed43.pt" \
    "$RESULTS/linear_probe_seed44.pt" \
    "$RESULTS/linear_probe_seed45.pt" | sed 's/,$//')

c2_probes=$(printf '"%s",' \
    "$RESULTS_C2/c2_linear_probe_seed42.pt" \
    "$RESULTS_C2/c2_linear_probe_seed43.pt" \
    "$RESULTS_C2/c2_linear_probe_seed44.pt" \
    "$RESULTS_C2/c2_linear_probe_seed45.pt" | sed 's/,$//')

c4_probes=$(printf '"%s",' \
    "$RESULTS_C4/c4_linear_probe_seed42.pt" \
    "$RESULTS_C4/c4_linear_probe_seed43.pt" \
    "$RESULTS_C4/c4_linear_probe_seed44.pt" \
    "$RESULTS_C4/c4_linear_probe_seed45.pt" | sed 's/,$//')

actsae_probes=$(printf '"%s",' \
    "$RESULTS_ACTSAE/actsae_linear_probe_seed42.pt" \
    "$RESULTS_ACTSAE/actsae_linear_probe_seed43.pt" \
    "$RESULTS_ACTSAE/actsae_linear_probe_seed44.pt" \
    "$RESULTS_ACTSAE/actsae_linear_probe_seed45.pt" | sed 's/,$//')

TRANS_CONFIG="$VIZ_OUT/transition_config.json"
cat > "$TRANS_CONFIG" <<JSON
{
  "n_samples_ms": 5000,
  "n_samples_pb": 0,
  "encodings": [
    {
      "label":        "Dense h_k",
      "ms_npz":       "${DENSE_EVAL_META}",
      "pb_npz":       "${PB_DENSE}",
      "probes":       [${dense_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "SSAE c=1",
      "ms_npz":       "${C1_EVAL_META}",
      "pb_npz":       "${PB_C1}",
      "probes":       [${c1_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "SSAE c=2 ReLU+L1",
      "ms_npz":       "${C2_EVAL_META}",
      "pb_npz":       "${PB_C2}",
      "probes":       [${c2_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "SSAE c=4 TopK*",
      "ms_npz":       "${C4_EVAL_META}",
      "pb_npz":       "${PB_C4}",
      "probes":       [${c4_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "Activation SAE",
      "ms_npz":       "${ACTSAE_EVAL_META}",
      "pb_npz":       "${PB_ACTSAE}",
      "probes":       [${actsae_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    }
  ]
}
JSON

python scripts/analyze_transition_probes.py \
    --config     "$TRANS_CONFIG" \
    --output-dir "$VIZ_OUT" \
    --dpi        150

echo ""
echo "=== All done ==="
echo ""
echo "  Figures:"
echo "    $VIZ_OUT/transition_comparison_ms.png"
echo "    $VIZ_OUT/transition_comparison_pb.png"
echo "    $VIZ_OUT/transition_r_summary_ms.png"
echo "    $VIZ_OUT/transition_r_summary_pb.png"
echo ""
echo "  Fetch locally:"
echo "  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_comparison/ ./results/mechanistic_comparison/"
