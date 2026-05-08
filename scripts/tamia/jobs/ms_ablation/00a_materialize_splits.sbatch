#!/bin/bash
#SBATCH --job-name=ms-abl-splits
#SBATCH --account=aip-azouaq
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 00a: Materialise Math-Shepherd split IDs.
#
# Runs once, no GPU.  Writes splits.json containing the explicit
# (solution_id, step_pos, label) tuples that all four model extraction jobs
# will load so every model encodes exactly the same examples.
#
# Output: $SCRATCH/cot-checker/ms_ablation/splits.json
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"
OUT_DIR="$SCRATCH_BASE/ms_ablation"

MS_DATA_FILE="$SCRATCH/cot-checker/ms_ablation/math_shepherd_gsm8k.jsonl"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "======================================================="
echo "  Stage 00a: Materialise splits"
echo "  Output: $OUT_DIR/splits.json"
echo "======================================================="

MS_ARG=""
[ -f "$MS_DATA_FILE" ] && MS_ARG="--ms-data-file $MS_DATA_FILE"

python scripts/model_size_ablation_materialize_splits.py \
    --output-dir   "$OUT_DIR" \
    --cache-dir    "$STORE/hf_cache" \
    --ms-per-class 10000 \
    --seed         42 \
    $MS_ARG

echo ""
echo "splits.json written to $OUT_DIR"
echo "Submit Stage 00 (extraction) once this job completes."
