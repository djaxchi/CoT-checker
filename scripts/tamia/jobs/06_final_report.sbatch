#!/bin/bash
#SBATCH --job-name=ptb-06-report
#SBATCH --account=aip-azouaq
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 06: Final report
#
# Reads summary_results.json and generates FINAL_REPORT.md with:
#   - Full benchmark table (all representations x all metrics)
#   - Explicit YES/NO answers to all 4 research questions with margins
#   - Per-PTB-variant validity summary
#
# Artifact: $PTB_PROBE_DIR/FINAL_REPORT.md
#
# Submit:
#   sbatch scripts/tamia/jobs/06_final_report.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
PTB_PROBE_DIR="$STORE/results/ptb_robust_probes"

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc python/3.11
source "$HOME/venvs/cot/bin/activate"

echo "======================================================="
echo "  Stage 06: Final report"
echo "======================================================="
echo ""

python scripts/ptb_final_report.py \
    --results "$PTB_PROBE_DIR/summary_results.json" \
    --out     "$PTB_PROBE_DIR/FINAL_REPORT.md"

echo ""
echo "  Content:"
echo "---"
cat "$PTB_PROBE_DIR/FINAL_REPORT.md"
echo "---"
echo ""
echo "======================================================="
echo "  Experiment complete."
echo ""
echo "  Fetch all results:"
echo "    rsync -avz $USER@tamia.alliancecan.ca:$PTB_PROBE_DIR/ ./results/ptb_robust_probes/"
echo "======================================================="
