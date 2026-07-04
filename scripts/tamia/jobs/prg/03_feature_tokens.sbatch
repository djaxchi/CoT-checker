#!/bin/bash
#SBATCH --job-name=prg-03-ftok
#SBATCH --account=aip-azouaq
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_geometry_v0 stage 3 (feature anatomy, per-token):
# per-token activations of the top SAE retrieval features over all direct
# prompts. Tiny job (one forward pass per prompt, ~220k tokens total); uses
# GPU 0 only, but TamIA allocates H100 nodes whole. Requires
# runs/parametric_retrieval_geometry_v0/sae/feature_pack_layer23.npz
# (committed) and the Qwen2.5-7B-Instruct HF cache.
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
if [[ -f slurm/s1_model_size/models.env ]]; then
    source slurm/s1_model_size/models.env
fi
export HF_HOME="${HF_CACHE_ROOT:-$STORE/hf_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 python scripts/parametric_retrieval/prg_feature_tokens.py \
    extract --out_dir "$RUN_DIR" --local_files_only
