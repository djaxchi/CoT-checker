#!/bin/bash
#SBATCH --job-name=ptb-05-pb
#SBATCH --account=aip-azouaq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 05: ProcessBench transfer evaluation
#
# (a) Extracts ProcessBench latents for all valid PTB variants + baselines.
# (b) Re-runs the probe benchmark with --reuse-probes (no retraining) and
#     --pb-dense to add PB-F1 and PB macro-F1 columns. Answers Q4.
#
# NOTE: Edit PTB_CKPTS below to include only variants where Q1=YES and
#       Q3 results are informative. By default all 4 are extracted.
#
# Artifacts:
#   $PTB_REP_DIR/*_pb.npz             -- PB latents
#   $PTB_PROBE_DIR/summary_results.json  -- updated with PB columns
#   $PTB_PROBE_DIR/summary_table.md      -- updated table
#
# Decision: read PB-F1 column. >25% = non-trivial transfer. See Q4 in log.
#
# Submit:
#   sbatch scripts/tamia/jobs/05_eval_processbench.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

PROBE_DATA="$STORE/probe_data"
PB_DIR="$SCRATCH_BASE/processbench"
PTB_CKPT_BASE="$STORE/results/checkpoints/ptb_robust"
PTB_REP_DIR="$SCRATCH_BASE/probe_data/ptb_representations_robust"
PTB_PROBE_DIR="$STORE/results/ptb_robust_probes"

cd "$PROJECT_DIR"
mkdir -p results/logs "$PTB_REP_DIR" "$PTB_PROBE_DIR"

module purge
module load StdEnv/2023 gcc python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export TOKENIZERS_PARALLELISM=false

echo "======================================================="
echo "  Stage 05: ProcessBench transfer"
echo "======================================================="
echo ""

# Build --ptb-checkpoints argument (edit to restrict to candidates of interest)
PTB_CKPTS=""
for name in no_l1 dwa_calibrated active_fraction topk; do
    ckpt="$PTB_CKPT_BASE/$name/best.pt"
    if [ -f "$ckpt" ]; then
        PTB_CKPTS="$PTB_CKPTS $ckpt:$name"
    fi
done

echo "--- (a) Extract ProcessBench latents ---"
python scripts/extract_predictive_transition_latents.py \
    --ptb-checkpoints $PTB_CKPTS \
    --ms-dense-eval   "$PROBE_DATA/dense_eval_held_out.npz" \
    --ms-dense-train  "$PROBE_DATA/dense_train_full.npz" \
    --pb-dense        "$PB_DIR/processbench_dense_gsm8k.npz" \
    --datasets        pb \
    --output-dir      "$PTB_REP_DIR" \
    --batch-size      4096 \
    --device          cpu

echo ""
echo "--- (b) Re-run probes with --reuse-probes + --pb-dense ---"
python scripts/eval_predictive_transition_probes.py \
    --rep-dir      "$PTB_REP_DIR" \
    --dense-train  "$PROBE_DATA/dense_train_full.npz" \
    --dense-eval   "$PROBE_DATA/dense_eval_held_out.npz" \
    --pb-dense     "$PB_DIR/processbench_dense_gsm8k.npz" \
    --output-dir   "$PTB_PROBE_DIR" \
    --epochs       50 \
    --device       cuda \
    --reuse-probes

echo ""
echo "======================================================="
echo "  Stage 05 complete."
echo "  Updated table : $PTB_PROBE_DIR/summary_table.md"
echo "  Full results  : $PTB_PROBE_DIR/summary_results.json"
echo ""
echo "  Fetch results:"
echo "    rsync -avz $USER@tamia.alliancecan.ca:$PTB_PROBE_DIR/ ./results/ptb_robust_probes/"
echo ""
echo "  Next: sbatch scripts/tamia/jobs/06_final_report.sbatch"
echo "======================================================="
