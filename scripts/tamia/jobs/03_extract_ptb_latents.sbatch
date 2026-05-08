#!/bin/bash
#SBATCH --job-name=ptb-03-extract
#SBATCH --account=aip-azouaq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 03: Extract MS latents (ms_train + ms_eval only, no ProcessBench yet)
#
# Produces one .npz per representation per MS split:
#   ptb_{name}_ms_train.npz / ptb_{name}_ms_eval.npz
#   dense_delta_ms_train.npz / dense_delta_ms_eval.npz
#   dense_concat_ms_train.npz / dense_concat_ms_eval.npz
#   random_bln_ms_train.npz / random_bln_ms_eval.npz
#
# dense_h is the input file itself; not re-generated.
# ProcessBench latents are extracted in Stage 05 after reviewing Stage 04.
#
# Artifact: $PTB_REP_DIR/*.npz
#
# Decision: inspect output inventory printed at end.
#   If any ptb_* file is missing, check validity_report.json for that variant.
#
# Submit:
#   sbatch scripts/tamia/jobs/03_extract_ptb_latents.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

PROBE_DATA="$STORE/probe_data"
PTB_CKPT_BASE="$STORE/results/checkpoints/ptb_robust"
PTB_REP_DIR="$SCRATCH_BASE/probe_data/ptb_representations_robust"

cd "$PROJECT_DIR"
mkdir -p results/logs "$PTB_REP_DIR"

module purge
module load StdEnv/2023 gcc python/3.11
source "$HOME/venvs/cot/bin/activate"

export TOKENIZERS_PARALLELISM=false

echo "======================================================="
echo "  Stage 03: Extract MS latents"
echo "======================================================="
echo ""

# Build --ptb-checkpoints argument from variants that have a checkpoint
PTB_CKPTS=""
for name in no_l1 dwa_calibrated active_fraction topk; do
    ckpt="$PTB_CKPT_BASE/$name/best.pt"
    if [ -f "$ckpt" ]; then
        PTB_CKPTS="$PTB_CKPTS $ckpt:$name"
        echo "  Using checkpoint: $name"
    else
        echo "  WARNING: checkpoint missing for $name, skipping."
    fi
done

if [ -z "$PTB_CKPTS" ]; then
    echo "ERROR: no valid PTB checkpoints found. Run Stage 01 first."
    exit 1
fi

python scripts/extract_predictive_transition_latents.py \
    --ptb-checkpoints $PTB_CKPTS \
    --ms-dense-eval   "$PROBE_DATA/dense_eval_held_out.npz" \
    --ms-dense-train  "$PROBE_DATA/dense_train_full.npz" \
    --datasets        ms_eval ms_train \
    --output-dir      "$PTB_REP_DIR" \
    --batch-size      4096 \
    --device          cpu

echo ""
echo "======================================================="
echo "  Stage 03 complete."
echo "  Representations: $PTB_REP_DIR/"
echo "  Next: sbatch scripts/tamia/jobs/04_eval_ms_probes.sbatch"
echo "======================================================="
