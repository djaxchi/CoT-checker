#!/bin/bash
#SBATCH --job-name=prg-07-patch
#SBATCH --account=aip-azouaq
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_geometry_v0 exp 1c: activation patching. Transplants
# each reasoning_unlocked question's post-CoT answer-onset residual (block 23,
# and 19) into its own direct-run decision point; matched vs mismatched-donor
# control. Needs only the committed metadata/grading + the Qwen HF cache.
# Single GPU used (node allocated whole).
# ---------------------------------------------------------------------------

set -euo pipefail
PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_geometry_v0}"

cd "$PROJECT_DIR"
mkdir -p results/logs
module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"
[[ -f slurm/s1_model_size/models.env ]] && source slurm/s1_model_size/models.env
export HF_HOME="${HF_CACHE_ROOT:-$STORE/hf_cache}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 python scripts/parametric_retrieval/prg_patch.py \
    --out_dir "$RUN_DIR" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only \
    --blocks 23 19 --alphas 1.0 0.5 --n 150
