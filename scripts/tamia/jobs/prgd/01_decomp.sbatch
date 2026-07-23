#!/bin/bash
#SBATCH --job-name=prgd-01-decomp
#SBATCH --account=aip-azouaq
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# ---------------------------------------------------------------------------
# parametric_retrieval_sae_decomp: offline SAE-feature decomposition of the
# flip (cheap go/no-go before feature steering). Encodes the test pairs' donor
# and recipient residuals at hs_idx 28 (= resid_post_layer_27) through the
# andyrdt L28 BatchTopK SAE and decomposes the difference. No model forward.
#
# Requires: the SAE downloaded to $SAE_ROOT (login node, see submit.sh), plus
# hidden_states_v1 (hs_idx 28) and pairs.parquet from access_v1.
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
RUN_DIR="${RUN_DIR:-runs/parametric_retrieval_access_v1}"

cd "$PROJECT_DIR"
mkdir -p results/logs

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2
source "$HOME/venvs/cot/bin/activate"

if [[ -f slurm/s1_model_size/models.env ]]; then
    source slurm/s1_model_size/models.env
fi
export HF_HOME="${HF_CACHE_ROOT:-$STORE/hf_cache}"
SAE_ROOT="${SAE_ROOT:-$HF_HOME/models/andyrdt/saes-qwen2.5-7b-instruct}"

CUDA_VISIBLE_DEVICES=0 python scripts/parametric_retrieval/prgd_sae_decomp.py \
    --out_dir "$RUN_DIR" --sae_root "$SAE_ROOT" --trainer "${TRAINER:-1}" \
    --n_pairs "${N_PAIRS:-800}"
echo "SAE DECOMPOSITION COMPLETE"
