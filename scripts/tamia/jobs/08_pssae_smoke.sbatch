#!/bin/bash
#SBATCH --job-name=pssae-08-smoke
#SBATCH --account=aip-azouaq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=128G
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot-checker"

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

COMMON_ARGS=(
    --backbone               "Qwen/Qwen2.5-0.5B"
    --hf-cache               "$STORE/hf_cache/hub"
    --train-json             "$STORE/data/gsm8k_385K_train.json"
    --valid-json             "$STORE/data/gsm8k_385K_valid.json"
    --pb-jsonl               "$SCRATCH_BASE/processbench/processbench_gsm8k.jsonl"
    --data-frac              0.5
    --max-train-transitions  400000
    --n-pb                   50
    --epochs                 5
    --batch-size             32
    --eval-batch-size        16
    --lr                     1e-3
    --max-seq-len            256
    --device                 cuda
)

# --- Variant A: with L1 (GPU 0) ---
CUDA_VISIBLE_DEVICES=0 python scripts/run_pssae_smoke.py \
    "${COMMON_ARGS[@]}" \
    --l1-coeff   0.01 \
    --output-dir "$STORE/results/pssae_l1" \
    > results/logs/pssae_l1-$SLURM_JOB_ID.out 2>&1 &
PID_L1=$!
echo "Started L1 variant (GPU 0)  PID=$PID_L1"

# --- Variant B: no L1 (GPU 1) ---
CUDA_VISIBLE_DEVICES=1 python scripts/run_pssae_smoke.py \
    "${COMMON_ARGS[@]}" \
    --l1-coeff   0.0 \
    --output-dir "$STORE/results/pssae_no_l1" \
    > results/logs/pssae_no_l1-$SLURM_JOB_ID.out 2>&1 &
PID_NO_L1=$!
echo "Started no-L1 variant (GPU 1)  PID=$PID_NO_L1"

# Wait for both; report exit codes
wait $PID_L1
STATUS_L1=$?
echo "L1 variant finished  exit=$STATUS_L1"

wait $PID_NO_L1
STATUS_NO_L1=$?
echo "No-L1 variant finished  exit=$STATUS_NO_L1"

if [ $STATUS_L1 -ne 0 ] || [ $STATUS_NO_L1 -ne 0 ]; then
    echo "One or both variants failed" >&2
    exit 1
fi
echo "Both variants completed successfully"
