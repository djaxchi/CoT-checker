#!/bin/bash
#SBATCH --job-name=ms-abl-probe
#SBATCH --account=aip-azouaq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 01: Train linear probes and evaluate on MS eval + ProcessBench.
#
# Runs two repr variants per model (raw and l2) so the report can determine
# whether L2 normalisation helps or hides a magnitude signal.  8 probe runs
# total (4 models × 2 reprs), all in parallel on CPU.
#
# No GPU required -- sklearn LogisticRegression.
#
# Reads from: $SCRATCH/cot-checker/ms_ablation/{ms_train,ms_eval,pb}_*.npz
# Writes to:  $SCRATCH/cot-checker/ms_ablation/probes/
#   metrics_{tag}_{repr}.json
#   scores_{tag}_{repr}.npz
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
SCRATCH_BASE="$SCRATCH/cot-checker"
DATA_DIR="$SCRATCH_BASE/ms_ablation"
OUT_DIR="$DATA_DIR/probes"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "======================================================="
echo "  Stage 01: Linear probes — raw + l2 variants (parallel)"
echo "  Data dir:   $DATA_DIR"
echo "  Output dir: $OUT_DIR"
echo "======================================================="

run_probe() {
    local tag=$1
    local repr=$2
    if [ ! -f "$DATA_DIR/ms_train_${tag}.npz" ]; then
        echo "  SKIP ${tag}/${repr}: ms_train_${tag}.npz not found"
        return 0
    fi
    echo "--- ${tag} repr=${repr} ---"
    python scripts/model_size_ablation_probe.py \
        --model-tag  "$tag" \
        --repr       "$repr" \
        --data-dir   "$DATA_DIR" \
        --output-dir "$OUT_DIR" \
        --seed       42 \
        --lr-C       1.0 \
        --max-iter   1000 \
        > "$OUT_DIR/probe_${tag}_${repr}.log" 2>&1 &
}

PIDS=()
for TAG in 0.5b 1.5b 3b 7b; do
    run_probe "$TAG" raw; PIDS+=($!)
    run_probe "$TAG" l2;  PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

echo ""
echo "======================================================="
if [ "$FAILED" -gt 0 ]; then
    echo "  FAILED: $FAILED probe run(s) failed."
    for TAG in 0.5b 1.5b 3b 7b; do
        for REPR in raw l2; do
            log="$OUT_DIR/probe_${TAG}_${REPR}.log"
            [ -f "$log" ] && echo "  --- ${TAG}/${REPR} ---" && tail -10 "$log"
        done
    done
    exit 1
fi

echo "  All probes complete."
echo ""
echo "  Results summary:"
for REPR in raw l2; do
    echo ""
    echo "  repr=$REPR:"
    for TAG in 0.5b 1.5b 3b 7b; do
        m="$OUT_DIR/metrics_${TAG}_${REPR}.json"
        [ -f "$m" ] && python3 -c "
import json
m = json.load(open('$m'))
ms = m['math_shepherd_eval']
pb = m['processbench']
col = ' [COLLAPSE]' if pb['collapse'] else ''
print(f\"    {m['model_tag']:6s}  MS AUROC={ms['auroc']:.4f}  MS F1={ms['macro_f1']:.4f}  \
PB AUROC={pb['auroc']:.4f}  PB F1={pb['macro_f1']:.4f}{col}\")
"
    done
done
echo "======================================================="
