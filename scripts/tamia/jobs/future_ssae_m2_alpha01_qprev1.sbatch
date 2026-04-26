#!/bin/bash
#SBATCH --job-name=fssae-m2-a01-qpv1
#SBATCH --account=aip-azouaq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
EXPNAME="future_ssae_m2_alpha01_qprev1"
OUTPUT_DIR="$STORE/results/checkpoints/$EXPNAME"

cd "$PROJECT_DIR"
mkdir -p results/logs "$OUTPUT_DIR"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== M2: q_prev1_current, alpha_pred=0.1 ==="
echo "Output: $OUTPUT_DIR"

python scripts/train_future_ssae_gsm8k.py \
    --data "$STORE/data/gsm8k_385K_train.json" \
    --val-data "$STORE/data/gsm8k_385K_valid.json" \
    --output-dir "$OUTPUT_DIR" \
    --model-id "Qwen/Qwen2.5-0.5B" \
    --alpha-pred 0.1 \
    --pred-context-mode q_prev1_current \
    --l1-target 3.0 \
    --epochs 2 \
    --batch-size 16 \
    --grad-accum 8 \
    --lr 1e-6 \
    --max-length 256 \
    --device cuda

echo "=== Done. Outputs in $OUTPUT_DIR ==="
ls -lh "$OUTPUT_DIR"
