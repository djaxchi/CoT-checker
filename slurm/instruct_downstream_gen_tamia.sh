#!/bin/bash
#SBATCH --job-name=instruct_downstream_gen
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Downstream TTS on Qwen3-8B Instruct, generation-state scoring (one pool per job).
#
# Every cell in CELLS_FILE (one checkpoint per leaderboard cell, the best seed)
# reads the states the sampler computed: one teacher-forced pass per trajectory,
# shared by all cells, sharded over the node's four GPUs. Scores land in
# $POOL/scores as <stem>__<cell>__gen.shard0<i>.jsonl beside the pool's existing
# confidence and PRM atoms, where the frontier reads them.
#
# Measured on this pool with one cell (job 492586): ~100 s of backbone per shard
# per dataset and ~5 s of head, so the hour is mostly margin.
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

: "${POOL:?set POOL to a pool directory}"
: "${CELLS_FILE:?set CELLS_FILE to a list of cell directories}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
export HF_HOME="$HF_CACHE"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$POOL/scores" "$POOL/logs"
LOG="$POOL/logs/downstream_gen-${SLURM_JOB_ID:-$$}.log"
mapfile -t CELLS < <(grep -v '^#' "$CELLS_FILE" | sed '/^\s*$/d')

cd "$PROJECT_ROOT"
echo "git_commit $(git rev-parse HEAD)  pool $POOL  cells ${#CELLS[@]}" | tee -a "$LOG"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy 2>&1 | tail -1

for STEM in tts_gsm8k tts_math500; do
  pids=()
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/score_gen_states_multi.py \
      --trajectories "$POOL"/"$STEM".shard*_trajectories.jsonl --cells "${CELLS[@]}" \
      --stem "$STEM" --out_dir "$POOL/scores" --model_name_or_path "$MODEL" \
      --local_files_only --layer 35 --shard_idx "$i" --num_shards 4 >>"$LOG" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] $STEM" >&2; tail -40 "$LOG" >&2; exit 1; }
  grep "span coverage" "$LOG" | tail -4
  echo "[ok] $STEM" | tee -a "$LOG"
done
echo "[$(date)] done" | tee -a "$LOG"
