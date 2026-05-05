#!/bin/bash
#SBATCH --job-name=cot-mech-c4
#SBATCH --account=aip-azouaq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/mechanistic_c4_%j.out
#SBATCH --error=logs/mechanistic_c4_%j.err

# ---------------------------------------------------------------------------
# Mechanistic analysis for c=4 TopK SSAE latents.
# CPU-only — reads pre-encoded .npz files and trained linear probe checkpoints.
#
# Submit: sbatch scripts/slurm/tamia_mechanistic_c4.sh
# Retrieve: rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_c4/ ./results/mechanistic_c4/
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
SCRATCH_RESULTS="$SCRATCH/cot-checker/results_c4"
EVAL_DATA="$SCRATCH/cot-checker/probe_data_c4/c4_eval_held_out.npz"
TRAIN_DATA="$SCRATCH/cot-checker/probe_data_c4/c4_train_full.npz"
OUTPUT_DIR="$PROJECT_DIR/results/mechanistic_c4"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11

source "$HOME/venvs/cot/bin/activate"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== Mechanistic analysis: c=4 TopK-40 SSAE ==="
echo "  Eval data  : $EVAL_DATA"
echo "  Train data : $TRAIN_DATA"
echo "  Probes     : $SCRATCH_RESULTS/c4_linear_probe_seed{42..45}.pt"
echo "  Output     : $OUTPUT_DIR"
echo ""

python scripts/mechanistic_analysis.py \
    --eval-data  "$EVAL_DATA" \
    --train-data "$TRAIN_DATA" \
    --probes     "$SCRATCH_RESULTS/c4_linear_probe_seed42.pt" \
                 "$SCRATCH_RESULTS/c4_linear_probe_seed43.pt" \
                 "$SCRATCH_RESULTS/c4_linear_probe_seed44.pt" \
                 "$SCRATCH_RESULTS/c4_linear_probe_seed45.pt" \
    --output-dir "$OUTPUT_DIR" \
    --label      "SSAE c=4 TopK-40" \
    --max-train-samples 50000

echo ""
echo "=== Done ==="
echo "Fetch plots with:"
echo "  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_c4/ ./results/mechanistic_c4/"
