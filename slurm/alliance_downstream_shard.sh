#!/bin/bash
#SBATCH --job-name=downstream_shard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# One shard of the Instruct downstream scoring on one GPU, for the Alliance
# clusters (Rorqual, Fir, Nibi) that allocate by GPU rather than by node. Many
# single-GPU jobs start through backfill far sooner than a whole node does.
#
#   TASK=gen  score every cell in CELLS_FILE from generation states (both datasets)
#   TASK=prm  score with Qwen2.5-Math-PRM-7B, segmentation gated against GATE_CELL
#
# Submit with --account; SHARD in 0..NUM_SHARDS-1. NO INTERNET on compute nodes.

set -euo pipefail
: "${POOL:?}" "${SHARD:?}" "${NUM_SHARDS:?}" "${TASK:?gen or prm}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$POOL/scores" "$POOL/logs"
LOG="$POOL/logs/${TASK}_shard${SHARD}-${SLURM_JOB_ID:-$$}.log"

cd "$PROJECT_ROOT"
echo "git $(git rev-parse --short HEAD) task $TASK pool $POOL shard $SHARD/$NUM_SHARDS" | tee -a "$LOG"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
if [[ "$TASK" == prm ]]; then
  pip install --no-index torch "transformers<5" numpy sympy 2>&1 | tail -1
else
  pip install --no-index torch transformers numpy sympy 2>&1 | tail -1
fi

for STEM in tts_gsm8k tts_math500; do
  if [[ "$TASK" == gen ]]; then
    : "${CELLS_FILE:?}"
    mapfile -t CELLS < <(grep -v '^#' "$CELLS_FILE" | sed '/^\s*$/d')
    python scripts/onpolicy/score_gen_states_multi.py \
      --trajectories "$POOL"/"$STEM".shard*_trajectories.jsonl --cells "${CELLS[@]}" \
      --stem "$STEM" --out_dir "$POOL/scores" --model_name_or_path Qwen/Qwen3-8B \
      --local_files_only --layer 35 --shard_idx "$SHARD" --num_shards "$NUM_SHARDS" 2>&1 | tee -a "$LOG" | tail -3
  else
    : "${GATE_CELL:?}"
    python scripts/onpolicy/score_traces_with_prm.py \
      --prm_name_or_path Qwen/Qwen2.5-Math-PRM-7B --local_files_only \
      --verify_against "$POOL"/scores/"$STEM"__"$GATE_CELL".shard*.jsonl \
      --trajectories "$POOL"/"$STEM".shard*_trajectories.jsonl \
      --shard_idx "$SHARD" --num_shards "$NUM_SHARDS" \
      --out "$POOL/scores/${STEM}__prm_qwen25_math_7b.shard${SHARD}.jsonl" 2>&1 | tee -a "$LOG" | tail -3
  fi
done
echo "[$(date)] done" | tee -a "$LOG"
