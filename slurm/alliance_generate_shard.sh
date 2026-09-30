#!/bin/bash
#SBATCH --job-name=gen_shard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# One shard of an offline TTS pool on one GPU (Alliance per-GPU clusters).
#   TASK=gen   Qwen3-8B writes N_SAMPLES solutions per problem (PROMPT_STYLE)
#   TASK=conf  token-confidence statistics for SHARD of all the pool's trajectories
# MAX_NEW_TOKENS is a safety bound only (default 16384): the run exists to have no
# cut-off, and every trace that reaches it is flagged hit_token_cap.
# NO INTERNET on compute nodes.

set -euo pipefail
: "${TASK:?gen or conf}" "${POOL:?}" "${STEM:?}" "${SHARD:?}" "${NUM_SHARDS:?}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$POOL"
cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) task $TASK stem $STEM shard $SHARD/$NUM_SHARDS"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy 2>&1 | tail -1

if [[ "$TASK" == gen ]]; then
  python scripts/generate_onpolicy_steps.py --fork_items "${PROBLEMS:?}" --id_field problem_id \
    --out_dir "$POOL" --stem "$STEM" --model_name_or_path Qwen/Qwen3-8B --local_files_only \
    --model_dtype bfloat16 --run_name "$STEM" --max_problems 0 --n_samples "${N_SAMPLES:-10}" \
    --temperature 1.0 --top_p 0.95 --top_k 50 --max_new_tokens "${MAX_NEW_TOKENS:-16384}" \
    --prompt_style "${PROMPT_STYLE:-prm800k}" --shard_idx "$SHARD" --num_shards "$NUM_SHARDS" --force
else
  python scripts/onpolicy/encode_token_confidence.py \
    --trajectories "$POOL"/"$STEM".shard*_trajectories.jsonl --out_dir "$POOL" --stem "$STEM" \
    --model_name_or_path Qwen/Qwen3-8B --local_files_only --model_dtype bfloat16 --topk 20 \
    --shard_idx "$SHARD" --num_shards "$NUM_SHARDS" --force
fi
