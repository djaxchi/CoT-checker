#!/bin/bash
#SBATCH --job-name=ms-abl-extract
#SBATCH --account=aip-azouaq
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gpus=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 00: Extract dense hidden states for all 4 Qwen2.5 model sizes.
#
# Prerequisite: Stage 00a (materialize_splits) must have completed and
# written $SCRATCH/cot-checker/ms_ablation/splits.json.  All four models
# load that file so they encode exactly the same steps.
#
# Hidden states are saved RAW (not L2-normalised).  L2 normalisation is
# applied at probe time via --repr {raw,l2} in Stage 01.
#
# GPU assignment:
#   GPU 0 → Qwen2.5-0.5B
#   GPU 1 → Qwen2.5-1.5B
#   GPU 2 → Qwen2.5-3B
#   GPU 3 → Qwen2.5-7B
#
# Approximate wall time: 0.5B~30min, 1.5B~60min, 3B~90min, 7B~180min.
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"
OUT_DIR="$SCRATCH_BASE/ms_ablation"
SPLIT_FILE="$OUT_DIR/splits.json"

# Optional local dataset files (leave empty to load from HF cache)
MS_DATA_FILE="$STORE/data/math_shepherd_gsm8k.jsonl"
PB_DATA_FILE="$SCRATCH_BASE/processbench/processbench_gsm8k.jsonl"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Build optional --ms-data-file and --pb-data-file args
MS_ARG=""
PB_ARG=""
[ -f "$MS_DATA_FILE" ] && MS_ARG="--ms-data-file $MS_DATA_FILE"
[ -f "$PB_DATA_FILE" ] && PB_ARG="--pb-data-file $PB_DATA_FILE"

if [ ! -f "$SPLIT_FILE" ]; then
    echo "ERROR: splits.json not found at $SPLIT_FILE"
    echo "Run Stage 00a (materialize_splits) first."
    exit 1
fi

echo "======================================================="
echo "  Stage 00: Extract hidden states (4 GPUs in parallel)"
echo "  Split file: $SPLIT_FILE"
echo "  Output dir: $OUT_DIR"
echo "======================================================="

run_extract() {
    local gpu=$1
    local model_id=$2
    local tag=$3
    local batch=$4
    echo ""
    echo "--- GPU $gpu: $model_id ($tag) ---"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/model_size_ablation_extract.py \
        --model-id    "$model_id" \
        --model-tag   "$tag" \
        --split-file  "$SPLIT_FILE" \
        --output-dir  "$OUT_DIR" \
        --cache-dir   "$STORE/hf_cache" \
        --batch-size  "$batch" \
        --max-seq-len 1024 \
        --dtype       float16 \
        --device      cuda \
        --seed        42 \
        $MS_ARG \
        $PB_ARG \
        > "$OUT_DIR/extract_${tag}.log" 2>&1 &
}

PIDS=()
run_extract 0 "Qwen/Qwen2.5-0.5B" "0.5b" 64;  PIDS+=($!)
run_extract 1 "Qwen/Qwen2.5-1.5B" "1.5b" 48;  PIDS+=($!)
run_extract 2 "Qwen/Qwen2.5-3B"   "3b"   32;  PIDS+=($!)
run_extract 3 "Qwen/Qwen2.5-7B"   "7b"   16;  PIDS+=($!)

TAGS=("0.5b" "1.5b" "3b" "7b")
FAILED=0
for i in "${!PIDS[@]}"; do
    tag="${TAGS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "  OK   $tag"
    else
        echo "  FAIL $tag  (see $OUT_DIR/extract_${tag}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "======================================================="
echo "  Summary: FAILED=$FAILED"
[ "$FAILED" -gt 0 ] && exit 1
echo "  All extractions complete."
echo "  Files in $OUT_DIR:"
ls -lh "$OUT_DIR"/*.npz 2>/dev/null || echo "  (no .npz files found)"
echo "======================================================="
