#!/bin/bash
#SBATCH --job-name=ptb-02-validity
#SBATCH --account=aip-azouaq
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 02: Validity report
#
# Reads validity_report.json + calibration.json for each of the 4 PTB
# variants and prints a compact decision table.
#
# Artifact: results/logs/ptb-02-validity-<JOBID>.out
#
# Decision rule (hard):
#   CONTINUE  if ptb_no_l1 has:
#               - cosine_sim  >= 0.05   (transition actually learned)
#               - R2          >= 0.0    (not worse than predicting the mean)
#               - z_mean      > 0       (encoder is active)
#               - no INVALID flags
#   STOP      otherwise -- debug training before extracting latents.
#
# Submit:
#   sbatch scripts/tamia/jobs/02_collect_ptb_validity.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
PTB_CKPT_BASE="$STORE/results/checkpoints/ptb_robust"

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc python/3.11
source "$HOME/venvs/cot/bin/activate"

echo "======================================================="
echo "  Stage 02: PTB Validity Report"
echo "======================================================="
echo ""

python scripts/ptb_validity_summary.py \
    --ckpt-base   "$PTB_CKPT_BASE" \
    --variants    no_l1 dwa_calibrated active_fraction topk \
    --cos-threshold 0.05 \
    --r2-threshold  0.0 \
    --sat-threshold 0.5

# ptb_validity_summary.py exits with code 1 if it recommends STOP,
# which will cause this job to fail and prevent downstream submission.
echo ""
echo "  Next: sbatch scripts/tamia/jobs/03_extract_ptb_latents.sbatch"
