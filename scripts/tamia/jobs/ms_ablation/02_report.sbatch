#!/bin/bash
#SBATCH --job-name=ms-abl-report
#SBATCH --account=aip-azouaq
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 02: Aggregate results, generate all figures, write final_report.md.
#
# Reads from:
#   $SCRATCH/cot-checker/ms_ablation/          -- raw .npz files (for PCA + delta plots)
#   $SCRATCH/cot-checker/ms_ablation/probes/   -- metrics_*.json + scores_*.npz
#
# Writes to: $HOME/CoT-checker/results/ms_ablation/
#   01_probe_histograms.png
#   02_pca_scatter.png
#   03_auroc_vs_size.png
#   04_macro_f1_vs_size.png
#   05_ppr_vs_size.png
#   06_confusion_matrices.png
#   07_delta_vs_weights.png
#   final_report.md
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"
DATA_DIR="$SCRATCH_BASE/ms_ablation"
SCORES_DIR="$DATA_DIR/probes"
REPORT_DIR="$PROJECT_DIR/results/ms_ablation"

cd "$PROJECT_DIR"
mkdir -p "$REPORT_DIR" results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

echo "======================================================="
echo "  Stage 02: Report generation"
echo "  Data dir:   $DATA_DIR"
echo "  Scores dir: $SCORES_DIR"
echo "  Output dir: $REPORT_DIR"
echo "======================================================="

python scripts/model_size_ablation_report.py \
    --data-dir   "$DATA_DIR" \
    --scores-dir "$SCORES_DIR" \
    --output-dir "$REPORT_DIR"

echo ""
echo "Files written to $REPORT_DIR:"
ls -lh "$REPORT_DIR"

echo ""
echo "To fetch results locally:"
echo "  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/ms_ablation/ ./results/ms_ablation/"
echo "======================================================="
