#!/bin/bash
# Submit the model-size ablation pipeline with SLURM job dependencies.
#
# Pipeline:
#   Stage 00a (CPU,  ~10m): materialise Math-Shepherd split IDs → splits.json
#   Stage 00  (GPU,  ~3h):  extract hidden states for all 4 model sizes
#   Stage 01  (CPU,  ~30m): probe raw + l2 variants (8 parallel runs)
#   Stage 02  (CPU,  ~15m): generate figures and final report
#
# Usage (from project root on Tamia):
#   bash scripts/tamia/jobs/ms_ablation/submit_pipeline.sh
#
# To skip stages that already completed:
#   bash scripts/tamia/jobs/ms_ablation/submit_pipeline.sh --skip-splits
#   bash scripts/tamia/jobs/ms_ablation/submit_pipeline.sh --skip-splits --skip-extract

set -euo pipefail

SKIP_SPLITS=false
SKIP_EXTRACT=false
for arg in "$@"; do
    [ "$arg" = "--skip-splits"  ] && SKIP_SPLITS=true
    [ "$arg" = "--skip-extract" ] && SKIP_EXTRACT=true
done

JOBS_DIR="scripts/tamia/jobs/ms_ablation"
mkdir -p results/logs

JID0A=""
JID0=""

if [ "$SKIP_SPLITS" = true ]; then
    echo "Skipping Stage 00a (--skip-splits)."
else
    JID0A=$(sbatch --parsable "$JOBS_DIR/00a_materialize_splits.sbatch")
    echo "Stage 00a (splits):  job $JID0A"
fi

if [ "$SKIP_EXTRACT" = true ]; then
    echo "Skipping Stage 00 (--skip-extract)."
else
    DEPEND_00=""
    [ -n "$JID0A" ] && DEPEND_00="--dependency=afterok:$JID0A"
    JID0=$(sbatch --parsable $DEPEND_00 "$JOBS_DIR/00_extract_all.sbatch")
    echo "Stage 00  (extract): job $JID0"
fi

DEPEND_01=""
[ -n "$JID0"  ] && DEPEND_01="--dependency=afterok:$JID0"
[ -n "$JID0A" ] && [ -z "$JID0" ] && DEPEND_01="--dependency=afterok:$JID0A"
JID1=$(sbatch --parsable $DEPEND_01 "$JOBS_DIR/01_probe_all.sbatch")
echo "Stage 01  (probes):  job $JID1"

JID2=$(sbatch --parsable --dependency=afterok:$JID1 "$JOBS_DIR/02_report.sbatch")
echo "Stage 02  (report):  job $JID2"

echo ""
echo "Monitor with:"
echo "  squeue -u $USER -o '%.10i %.9P %.30j %.8T %.10M %.6D %R'"
[ -n "$JID0A" ] && echo "" && echo "Stream splits log:" && \
    echo "  tail -f results/logs/ms-abl-splits-${JID0A}.out"
[ -n "$JID0" ]  && echo "" && echo "Stream extract log:" && \
    echo "  tail -f results/logs/ms-abl-extract-${JID0}.out"
