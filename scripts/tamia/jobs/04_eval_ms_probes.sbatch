#!/bin/bash
#SBATCH --job-name=ptb-04-ms-probes
#SBATCH --account=aip-azouaq
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 04: Math-Shepherd linear probe benchmark (no ProcessBench)
#
# Trains and evaluates probes on all available representations for the
# Math-Shepherd correctness task only. Answers Q1, Q2, Q3.
#
# Flags: --controls (randomization controls) + --balance-sweep
# No --pb-dense: ProcessBench evaluation is deferred to Stage 05.
#
# Artifacts:
#   $PTB_PROBE_DIR/summary_results.json  -- full results
#   $PTB_PROBE_DIR/summary_table.md      -- benchmark table
#   $PTB_PROBE_DIR/*_linear_probe_seed*.pt -- saved probe weights (for Stage 05)
#
# Decision: read summary_table.md and the Q1/Q2/Q3 answers in the log.
#   If ptb_no_l1 does not beat random_bln (Q1 = NO), investigate before
#   running Stage 05 -- ProcessBench would not add useful signal.
#
# Submit:
#   sbatch scripts/tamia/jobs/04_eval_ms_probes.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

PROBE_DATA="$STORE/probe_data"
PTB_REP_DIR="$SCRATCH_BASE/probe_data/ptb_representations_robust"
PTB_PROBE_DIR="$STORE/results/ptb_robust_probes"

cd "$PROJECT_DIR"
mkdir -p results/logs "$PTB_PROBE_DIR"

module purge
module load StdEnv/2023 gcc python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export TOKENIZERS_PARALLELISM=false

echo "======================================================="
echo "  Stage 04: Math-Shepherd probe benchmark"
echo "======================================================="
echo ""

python scripts/eval_predictive_transition_probes.py \
    --rep-dir      "$PTB_REP_DIR" \
    --dense-train  "$PROBE_DATA/dense_train_full.npz" \
    --dense-eval   "$PROBE_DATA/dense_eval_held_out.npz" \
    --output-dir   "$PTB_PROBE_DIR" \
    --epochs       50 \
    --device       cuda \
    --controls \
    --balance-sweep

echo ""
echo "======================================================="
echo "  Stage 04 complete."
echo "  Benchmark table : $PTB_PROBE_DIR/summary_table.md"
echo "  Full results    : $PTB_PROBE_DIR/summary_results.json"
echo ""
echo "  Review Q1/Q2/Q3 in the log above before proceeding."
echo "  If Q1=NO (ptb_no_l1 <= random_bln): stop and debug."
echo ""
echo "  Fetch results:"
echo "    rsync -avz $USER@tamia.alliancecan.ca:$PTB_PROBE_DIR/summary_table.md ./results/ptb_robust_probes/"
echo ""
echo "  Next (if Q1=YES): sbatch scripts/tamia/jobs/05_eval_processbench.sbatch"
echo "======================================================="
