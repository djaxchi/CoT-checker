#!/bin/bash
#SBATCH --job-name=ptb-07-eval
#SBATCH --account=aip-azouaq
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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

python scripts/eval_ptb_vs_probe.py \
    --data-file   "$SCRATCH_BASE/processbench/processbench_gsm8k.jsonl" \
    --ssae-ckpt   "$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt" \
    --ptb-ckpt    "$STORE/results/checkpoints/ptb_robust/no_l1/best.pt" \
    --probe-ckpt  "$STORE/results/ptb_robust_probes/dense_h_linear_probe_seed42.pt" \
    --n-solutions 50 \
    --device      cpu
