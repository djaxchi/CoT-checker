#!/bin/bash
#SBATCH --job-name=ptb-06-pb-full
#SBATCH --account=aip-azouaq
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# Stage 06: Re-encode ProcessBench GSM8K with full solutions (no truncation).
#
# Standard encoding stops at the first-error step (label+1), which means
# h_{k+1} never exists for the error step and PTB reconstruction-error
# anomaly detection cannot score it.  This job encodes ALL steps so that
# h_{k+1} is available at the error step for 92.3% of incorrect solutions.
#
# Output:
#   $SCRATCH/cot-checker/processbench/processbench_dense_gsm8k_full.npz
#
# Submit:
#   sbatch scripts/tamia/jobs/06_encode_pb_full_solution.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

CKPT_SSAE="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"
PB_JSONL="$SCRATCH_BASE/processbench/processbench_gsm8k.jsonl"
OUT_NPZ="$SCRATCH_BASE/processbench/processbench_dense_gsm8k_full.npz"

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "======================================================="
echo "  Stage 06: Encode PB GSM8K full solutions (dense h_k)"
echo "======================================================="
echo ""

if [ -f "$OUT_NPZ" ]; then
    echo "  Already exists: $OUT_NPZ"
    echo "  Delete it to re-run."
    exit 0
fi

CUDA_VISIBLE_DEVICES=0 python scripts/encode_processbench.py \
    --checkpoint  "$CKPT_SSAE" \
    --encoding    dense \
    --full-solution \
    --data-file   "$PB_JSONL" \
    --output      "$OUT_NPZ" \
    --batch-size  32 \
    --device      cuda

echo ""
echo "======================================================="
echo "  Stage 06 complete."
echo "  Output: $OUT_NPZ"
echo ""
echo "  Next: run PTB anomaly detection on full solutions"
echo "    python scripts/eval_ptb_anomaly.py \\"
echo "        --pb-dense   $OUT_NPZ \\"
echo "        --ptb-ckpt   $STORE/results/checkpoints/ptb_robust/no_l1/best.pt \\"
echo "        --n-solutions 207"
echo "======================================================="
