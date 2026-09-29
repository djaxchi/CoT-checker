#!/bin/bash
#SBATCH --job-name=online_shard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# online_reject_v2: one arm, one shard of one problem set, one GPU (Alliance
# per-GPU clusters). Qwen3-8B writes each step; the panel (two generation-state
# probes and Qwen2.5-Math-PRM-7B) scores every draft, and ACTIVE decides.
#
#   ARM=plain        no checker decision; every step scored by the whole panel
#   ARM=reject       resample a step whose ACTIVE suspicion exceeds TAU (2 retries)
#   ARM=reject_blind the same loop with a coin at BLIND_RATE; drafts still scored
#   ARM=verify       gate: online scores reproduce the offline ones on stored traces
#
# NO INTERNET on compute nodes; models are read from HF_HOME.

set -euo pipefail
: "${ARM:?}" "${OUT:?}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CELLS_DIR="${CELLS_DIR:-$SCRATCH/cot_mech/cells}"
GEN_CELLS=("$CELLS_DIR/step_stats__mlp_h1024__seed43" "$CELLS_DIR/boundary_stats__mlp_h1024x2__seed42")
PRM=Qwen/Qwen2.5-Math-PRM-7B
mkdir -p "$(dirname "$OUT")"

cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) arm $ARM active ${ACTIVE:-none} tau ${TAU:-} shard ${SHARD:-0}/${NUM_SHARDS:-1}"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
# transformers < 5: the PRM's remote code needs it, and 4.x runs Qwen3.
pip install --no-index torch "transformers<5" numpy sympy 2>&1 | tail -1

if [[ "$ARM" == verify ]]; then
  python scripts/onpolicy/verify_online_checkers.py --pool "${VERIFY_POOL:?}" \
    --gen_cells "${GEN_CELLS[@]}" --prm_name_or_path "$PRM" | tee "$OUT"
  exit "${PIPESTATUS[0]}"
fi

EXTRA=()
[[ "$ARM" == reject ]] && EXTRA+=(--reject_tau "${TAU:?}")
[[ "$ARM" == reject_blind ]] && EXTRA+=(--reject_tau 1.0 --blind_retry_rate "${BLIND_RATE:?}" --score_all_drafts)
[[ "$ARM" == plain ]] && EXTRA+=(--score_plain)

python scripts/onpolicy/online_bon.py --checker panel \
  --gen_cells "${GEN_CELLS[@]}" --prm_name_or_path "$PRM" --active "${ACTIVE:-none}" \
  --traces "${PROBLEMS:?}" --model_name_or_path Qwen/Qwen3-8B --local_files_only \
  --layer 35 --prompt_style chat --arms "$ARM" \
  --plain_temperature 1.0 --reject_temperature 1.0 --top_p 0.95 --top_k 50 \
  --max_steps 28 --max_new_tokens 768 --max_retries 2 --max_problems 100000 \
  --seed 42 --shard_idx "${SHARD:-0}" --num_shards "${NUM_SHARDS:-1}" \
  --out "$OUT" "${EXTRA[@]}"
