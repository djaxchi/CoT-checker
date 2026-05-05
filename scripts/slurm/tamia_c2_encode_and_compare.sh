#!/bin/bash
#SBATCH --job-name=cot-c2-compare
#SBATCH --account=aip-azouaq
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:4
#SBATCH --output=logs/c2_compare_%j.out
#SBATCH --error=logs/c2_compare_%j.err

# ---------------------------------------------------------------------------
# Encode ProcessBench with SSAE c=2 and regenerate the 5-encoding
# mechanistic comparison figures (adds c=2 ReLU+L1 column).
#
# Run AFTER tamia_train_ssae_c2.sh has completed.
#
# Produces updated figures in ~/CoT-checker/results/mechanistic_comparison/:
#   comparison_ms.png   — Math-Shepherd: Dense | c=1 | c=2 ReLU+L1 | c=4 TopK* | ActSAE
#   comparison_pb.png   — ProcessBench: same columns
#   comparison_ms_r_summary.png
#   comparison_pb_r_summary.png
#
# Submit: sbatch scripts/slurm/tamia_c2_encode_and_compare.sh
# Fetch:  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_comparison/ ./results/mechanistic_comparison/
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

CKPT_C2="$STORE/checkpoints/ssae_c2_relu_l1.pt"
PB_JSONL="$SCRATCH_BASE/processbench/processbench_gsm8k.jsonl"
PB_C2="$SCRATCH_BASE/processbench/processbench_c2_gsm8k.npz"
VIZ_OUT="$PROJECT_DIR/results/mechanistic_comparison"

cd "$PROJECT_DIR"
mkdir -p logs "$VIZ_OUT"

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
for f in "$CKPT_C2" "$PB_JSONL"; do
    [ ! -f "$f" ] && { echo "ERROR: $f not found. Did tamia_train_ssae_c2.sh finish?"; exit 1; }
done

for seed in 42 43 44 45; do
    f="$STORE/results_c2/c2_linear_probe_seed${seed}.pt"
    [ ! -f "$f" ] && { echo "ERROR: $f missing."; exit 1; }
done
echo "  All required files found."
echo ""

# ---------------------------------------------------------------------------
# Encode ProcessBench with SSAE c=2
# ---------------------------------------------------------------------------
echo "=== Encoding ProcessBench with SSAE c=2 ==="
CUDA_VISIBLE_DEVICES=0 python scripts/encode_processbench.py \
    --checkpoint "$CKPT_C2" \
    --data-file  "$PB_JSONL" \
    --output     "$PB_C2" \
    --encoding   ssae \
    --batch-size 32 \
    --max-seq-len 2048 \
    --device     cuda
echo ""

# ---------------------------------------------------------------------------
# Build probe path lists
# ---------------------------------------------------------------------------
e1_probes=$(printf '"%s",' $(for s in 42 43 44 45; do echo "$STORE/results/dense_linear_probe_seed${s}.pt"; done) | sed 's/,$//')
e2_probes=$(printf '"%s",' $(for s in 42 43 44 45; do echo "$STORE/results/linear_probe_seed${s}.pt"; done) | sed 's/,$//')
e3_probes=$(printf '"%s",' $(for s in 42 43 44 45; do
    f="$STORE/results_c4/c4_linear_probe_seed${s}.pt"
    [ -f "$f" ] || f="$SCRATCH_BASE/results_c4/c4_linear_probe_seed${s}.pt"
    echo "$f"
done) | sed 's/,$//')
e4_probes=$(printf '"%s",' $(for s in 42 43 44 45; do echo "$STORE/results_c2/c2_linear_probe_seed${s}.pt"; done) | sed 's/,$//')
e5_probes=$(printf '"%s",' $(for s in 42 43 44 45; do echo "$SCRATCH_BASE/results_actsae/actsae_linear_probe_seed${s}.pt"; done) | sed 's/,$//')

# ---------------------------------------------------------------------------
# Generate updated 5-encoding config JSON
# ---------------------------------------------------------------------------
CONFIG_JSON="$VIZ_OUT/config.json"
cat > "$CONFIG_JSON" <<JSON
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
      "ms_npz":       "${SCRATCH_BASE}/probe_data_c2/c2_eval_held_out.npz",
      "pb_npz":       "${PB_C2}",
      "probes":       [${e4_probes}],
      "ms_label_key": "correctness",
      "pb_label_key": "step_labels"
    },
    {
      "label":        "SSAE c=4 TopK*",
      "ms_npz":       "${SCRATCH_BASE}/probe_data_c4/c4_eval_held_out.npz",
      "pb_npz":       "${SCRATCH_BASE}/processbench/processbench_c4_gsm8k.npz",
      "probes":       [${e3_probes}],
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

echo "Config written: $CONFIG_JSON"
echo ""

# ---------------------------------------------------------------------------
# Generate comparison figures
# ---------------------------------------------------------------------------
echo "=== Generating 5-encoding comparison figures ==="
python scripts/compare_mechanistic_viz.py \
    --config     "$CONFIG_JSON" \
    --output-dir "$VIZ_OUT" \
    --reducer    pca \
    --dpi        150

echo ""
echo "=== Done ==="
echo "  Figures in: $VIZ_OUT"
echo ""
echo "  Fetch locally with:"
echo "  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_comparison/ ./results/mechanistic_comparison/"
