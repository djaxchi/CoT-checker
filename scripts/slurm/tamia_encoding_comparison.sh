#!/bin/bash
#SBATCH --job-name=cot-enc-compare
#SBATCH --account=aip-azouaq
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:2
#SBATCH --output=logs/encoding_comparison_%j.out
#SBATCH --error=logs/encoding_comparison_%j.err

# ---------------------------------------------------------------------------
# 4-way encoding comparison: Dense h_k | SSAE c=1 | SSAE c=4 | Activation SAE
#
# Produces multi-panel mechanistic comparison figures for Math-Shepherd and
# ProcessBench, then saves them to ~/CoT-checker/results/mechanistic_comparison/.
#
# GPU-optimised layout (2 H100s, 3h walltime):
#   Phase 1  [parallel, ~15 min]
#     GPU 0   encode ProcessBench with dense backbone (E1)
#     GPU 1   encode ProcessBench with SSAE c=4 (E3)
#     CPU     train activation SAE on dense_train_full.npz (E4)
#              + encode MS train/eval through SAE
#              + train 4 linear probes for E4
#   Phase 2  [serial, ~1 min]
#     CPU     apply activation SAE to ProcessBench dense npz (E4 PB)
#   Phase 3  [serial, ~3 min]
#     CPU     generate comparison viz JSON config
#             run compare_mechanistic_viz.py for both datasets
#
# Prerequisites:
#   - SSAE c=1 checkpoint in $STORE/checkpoints/
#   - SSAE c=4 checkpoint in $STORE/checkpoints/
#   - E2 ProcessBench gsm8k npz already created (tamia_processbench.sh)
#   - Dense MS train/eval npz already created (tamia_dense_ablation.sh)
#   - Dense linear probes for E1 in $STORE/results/
#   - SSAE c=1 linear probes for E2 in $STORE/results/
#   - SSAE c=4 linear probes for E3 in $STORE/results_c4/ or $SCRATCH/cot-checker/results_c4/
#   - ProcessBench JSONL exported to $SCRATCH/cot-checker/processbench/
#
# Submit: sbatch scripts/slurm/tamia_encoding_comparison.sh
# Fetch:  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_comparison/ ./results/mechanistic_comparison/
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

# --- Paths ---
SCRATCH_DATA="$SCRATCH_BASE/probe_data"
SCRATCH_C4DATA="$SCRATCH_BASE/probe_data_c4"
SCRATCH_RESULTS="$SCRATCH_BASE/results"
SCRATCH_C4RESULTS="$SCRATCH_BASE/results_c4"
SCRATCH_PB="$SCRATCH_BASE/processbench"
SCRATCH_ACTSAE="$SCRATCH_BASE/results_actsae"

VIZ_OUT="$PROJECT_DIR/results/mechanistic_comparison"

CKPT_C1="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"
CKPT_C4="$STORE/checkpoints/ssae_c4_topk40_frozen.pt"

# Math-Shepherd eval .npz (one per encoding)
MS_EVAL_E1="$SCRATCH_DATA/dense_eval_held_out.npz"
MS_EVAL_E2="$SCRATCH_DATA/eval_held_out.npz"
MS_EVAL_E3="$SCRATCH_C4DATA/c4_eval_held_out.npz"
MS_EVAL_E4="$SCRATCH_ACTSAE/actsae_eval_held_out.npz"

# Math-Shepherd training .npz for E4 activation SAE
MS_TRAIN_E1="$SCRATCH_DATA/dense_train_full.npz"

# ProcessBench encoded .npz files
PB_JSONL="$SCRATCH_PB/processbench_gsm8k.jsonl"     # pre-exported
PB_E2="$SCRATCH_PB/processbench_gsm8k.npz"           # SSAE c=1 — already created
PB_E1="$SCRATCH_PB/processbench_dense_gsm8k.npz"     # dense — to create
PB_E3="$SCRATCH_PB/processbench_c4_gsm8k.npz"        # SSAE c=4 — to create
PB_E4="$SCRATCH_ACTSAE/processbench_actsae_gsm8k.npz" # activation SAE — to create

# Linear probe checkpoints (one path per seed, 4 seeds)
PROBES_E1=()
PROBES_E2=()
PROBES_E3=()
for seed in 42 43 44 45; do
    PROBES_E1+=("$STORE/results/dense_linear_probe_seed${seed}.pt")
    PROBES_E2+=("$STORE/results/linear_probe_seed${seed}.pt")
    # c4 probes: check both possible locations
    C4_PROBE="$STORE/results_c4/c4_linear_probe_seed${seed}.pt"
    if [ ! -f "$C4_PROBE" ]; then
        C4_PROBE="$SCRATCH_C4RESULTS/c4_linear_probe_seed${seed}.pt"
    fi
    PROBES_E3+=("$C4_PROBE")
done
PROBES_E4=()
for seed in 42 43 44 45; do
    PROBES_E4+=("$SCRATCH_ACTSAE/actsae_linear_probe_seed${seed}.pt")
done

# ---

cd "$PROJECT_DIR"
mkdir -p logs "$SCRATCH_ACTSAE" "$SCRATCH_PB" "$VIZ_OUT"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2

source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
echo "=== Pre-flight checks ==="
for f in "$CKPT_C1" "$CKPT_C4" "$MS_EVAL_E1" "$MS_EVAL_E2" "$MS_EVAL_E3" \
          "$MS_TRAIN_E1" "$PB_JSONL"; do
    if [ ! -f "$f" ]; then
        echo "  ERROR: required file not found: $f"
        exit 1
    fi
done

if [ ! -f "$PB_E2" ]; then
    echo "  ERROR: ProcessBench SSAE c=1 encoding not found: $PB_E2"
    echo "  Run tamia_processbench.sh first."
    exit 1
fi

MISSING_PROBES=0
for p in "${PROBES_E1[@]}" "${PROBES_E2[@]}"; do
    [ ! -f "$p" ] && { echo "  WARNING: probe not found: $p"; MISSING_PROBES=1; }
done
for p in "${PROBES_E3[@]}"; do
    [ ! -f "$p" ] && { echo "  WARNING: c=4 probe not found: $p"; MISSING_PROBES=1; }
done
[ "$MISSING_PROBES" -eq 1 ] && echo "  (missing probes will be skipped in viz)"
echo "  All required files found."
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Parallel — GPU encoding + CPU SAE training
# ---------------------------------------------------------------------------
echo "=== [Phase 1] Parallel encoding + SAE training ==="

# GPU 0: encode ProcessBench with dense backbone (E1)
echo "  GPU 0: Encoding ProcessBench dense (E1)..."
CUDA_VISIBLE_DEVICES=0 python scripts/encode_processbench.py \
    --checkpoint "$CKPT_C1" \
    --data-file  "$PB_JSONL" \
    --output     "$PB_E1" \
    --encoding   dense \
    --batch-size 32 \
    --max-seq-len 2048 \
    --device     cuda \
    > "$SCRATCH_PB/encode_dense.log" 2>&1 &
PID_DENSE=$!

# GPU 1: encode ProcessBench with SSAE c=4 (E3)
echo "  GPU 1: Encoding ProcessBench SSAE c=4 (E3)..."
CUDA_VISIBLE_DEVICES=1 python scripts/encode_processbench.py \
    --checkpoint "$CKPT_C4" \
    --data-file  "$PB_JSONL" \
    --output     "$PB_E3" \
    --encoding   ssae \
    --batch-size 32 \
    --max-seq-len 2048 \
    --device     cuda \
    > "$SCRATCH_PB/encode_c4.log" 2>&1 &
PID_C4=$!

# CPU: train activation SAE + encode MS train/eval + train linear probes (E4)
echo "  CPU: Training activation SAE (E4)..."
python scripts/train_activation_sae.py \
    --train-data  "$MS_TRAIN_E1" \
    --eval-data   "$MS_EVAL_E1" \
    --output-dir  "$SCRATCH_ACTSAE" \
    --n-latents   896 \
    --k           40 \
    --epochs      20 \
    --batch-size  2048 \
    --probe-epochs 50 \
    --device      cpu \
    > "$SCRATCH_ACTSAE/train_actsae.log" 2>&1 &
PID_SAE=$!

# Wait for all three
FAILED=0
for pid_var in PID_DENSE PID_C4 PID_SAE; do
    eval pid=\$$pid_var
    tag=$(echo $pid_var | sed 's/PID_//')
    if wait "$pid"; then
        echo "  $tag: finished OK"
    else
        echo "  ERROR: $tag failed"
        FAILED=$((FAILED + 1))
    fi
done
echo ""
cat "$SCRATCH_PB/encode_dense.log" | tail -5
cat "$SCRATCH_PB/encode_c4.log"    | tail -5
cat "$SCRATCH_ACTSAE/train_actsae.log" | tail -10
echo ""

[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED phase-1 job(s) failed. Aborting."; exit 1; }

# ---------------------------------------------------------------------------
# Phase 2: Apply activation SAE to ProcessBench dense npz (fast, CPU)
# ---------------------------------------------------------------------------
echo "=== [Phase 2] Encoding ProcessBench with activation SAE (E4) ==="
python scripts/train_activation_sae.py \
    --encode-only \
    --actsae-checkpoint "$SCRATCH_ACTSAE/actsae.pt" \
    --encode-extra      "${PB_E1}:${PB_E4}" \
    --device            cpu
echo ""

# ---------------------------------------------------------------------------
# Phase 3: Generate config JSON and run comparison viz
# ---------------------------------------------------------------------------
echo "=== [Phase 3] Generating comparison figures ==="

CONFIG_JSON="$VIZ_OUT/config.json"

# Build probe path arrays as JSON strings
e1_probes=$(printf '"%s",' "${PROBES_E1[@]}" | sed 's/,$//')
e2_probes=$(printf '"%s",' "${PROBES_E2[@]}" | sed 's/,$//')
e3_probes=$(printf '"%s",' "${PROBES_E3[@]}" | sed 's/,$//')
e4_probes=$(printf '"%s",' "${PROBES_E4[@]}" | sed 's/,$//')

cat > "$CONFIG_JSON" <<JSON
{
  "n_samples_ms": 5000,
  "n_samples_pb": 0,
  "encodings": [
    {
      "label":        "Dense h_k",
      "ms_npz":       "${MS_EVAL_E1}",
      "pb_npz":       "${PB_E1}",
      "probes":       [${e1_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "SSAE c=1",
      "ms_npz":       "${MS_EVAL_E2}",
      "pb_npz":       "${PB_E2}",
      "probes":       [${e2_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "SSAE c=4 TopK",
      "ms_npz":       "${MS_EVAL_E3}",
      "pb_npz":       "${PB_E3}",
      "probes":       [${e3_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "Activation SAE",
      "ms_npz":       "${MS_EVAL_E4}",
      "pb_npz":       "${PB_E4}",
      "probes":       [${e4_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    }
  ]
}
JSON

echo "  Config written: $CONFIG_JSON"

python scripts/compare_mechanistic_viz.py \
    --config     "$CONFIG_JSON" \
    --output-dir "$VIZ_OUT" \
    --reducer    pca \
    --dpi        150

echo ""
echo "=== Done ==="
echo ""
echo "  Figures:"
echo "    $VIZ_OUT/comparison_ms.png"
echo "    $VIZ_OUT/comparison_ms_r_summary.png"
echo "    $VIZ_OUT/comparison_pb.png"
echo "    $VIZ_OUT/comparison_pb_r_summary.png"
echo ""
echo "  Fetch locally with:"
echo "    rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_comparison/ ./results/mechanistic_comparison/"
