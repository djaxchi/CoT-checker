#!/bin/bash
#SBATCH --job-name=cot-mechanistic
#SBATCH --account=aip-azouaq
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/mechanistic_%j.out
#SBATCH --error=logs/mechanistic_%j.err

# ---------------------------------------------------------------------------
# Mechanistic analysis of SSAE latents from the experiment-7 encodings.
#
# CPU-only job: reads pre-encoded .npz files and trained linear probe
# checkpoints from $STORE, produces plots and a textual summary.
#
# Prerequisites:
#   - eval_held_out.npz in $STORE/probe_data/
#   - train_final.npz in $STORE/probe_data/
#   - linear_probe_seed{42..45}.pt in $STORE/results/
#
# Submit: sbatch scripts/slurm/tamia_mechanistic.sh
# Retrieve plots: rsync -avz $USER@tamia.calculquebec.ca:~/CoT-checker/results/mechanistic/ ./results/mechanistic/
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"

EVAL_DATA="$STORE/probe_data/eval_held_out.npz"
TRAIN_DATA="$STORE/probe_data/train_final.npz"
RESULTS_DIR="$STORE/results"
OUTPUT_DIR="$PROJECT_DIR/results/mechanistic"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11

source "$HOME/venvs/cot/bin/activate"

# Ensure analysis dependencies are present (safe no-op if already installed)
pip install --quiet matplotlib 2>/dev/null || true

# Offline mode: no HF downloads needed for this analysis
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== Mechanistic analysis ==="
echo "  Eval data  : $EVAL_DATA"
echo "  Train data : $TRAIN_DATA"
echo "  Probes     : $RESULTS_DIR/linear_probe_seed{42..45}.pt"
echo "  Output     : $OUTPUT_DIR"
echo ""

python scripts/mechanistic_analysis.py \
    --eval-data  "$EVAL_DATA" \
    --train-data "$TRAIN_DATA" \
    --probes     "$RESULTS_DIR/linear_probe_seed42.pt" \
                 "$RESULTS_DIR/linear_probe_seed43.pt" \
                 "$RESULTS_DIR/linear_probe_seed44.pt" \
                 "$RESULTS_DIR/linear_probe_seed45.pt" \
    --output-dir "$OUTPUT_DIR" \
    --max-train-samples 50000

echo ""
echo "=== Copying plots to project space ==="
ls -lh "$OUTPUT_DIR"

echo ""
echo "=== Done ==="
echo "Fetch results with:"
echo "  rsync -avz $USER@tamia.calculquebec.ca:~/CoT-checker/results/mechanistic/ ./results/mechanistic/"
