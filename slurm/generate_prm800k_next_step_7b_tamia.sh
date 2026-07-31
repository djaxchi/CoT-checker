#!/bin/bash
#SBATCH --job-name=gen_next_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Generate the next step (W=1) for PRM800K splits with Qwen2.5-7B base, to give the
# future-delta (pcd) representation a materialized continuation to train on. Set
# LIMIT for a smoke (small N, one GPU) or leave empty for the full split (4-GPU
# in-node shards, merged after). IN_STEM picks the split.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
DATA_DIR="$RUN_ROOT/data"
OUT_DIR="$RUN_ROOT/data_pcd"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
IN_STEM="${IN_STEM:-prm800k_test_2k}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NEW="${MAX_NEW:-64}"
LIMIT="${LIMIT:-}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "$OUT_DIR"
export HF_HOME="$HF_CACHE"; export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

IN_JSONL="$DATA_DIR/${IN_STEM}.jsonl"
LIM_ARG=""; [[ -n "$LIMIT" ]] && LIM_ARG="--limit $LIMIT"

if [[ -n "$LIMIT" ]]; then
  # smoke: single shard, one GPU
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_prm800k_next_step.py \
    --in_jsonl "$IN_JSONL" --out_jsonl "$OUT_DIR/${IN_STEM}_next.jsonl" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --max_new_tokens "$MAX_NEW" --batch_size "$BATCH_SIZE" $LIM_ARG
else
  pids=()
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_prm800k_next_step.py \
      --in_jsonl "$IN_JSONL" --out_jsonl "$OUT_DIR/${IN_STEM}_next.shard${i}.jsonl" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
      --max_new_tokens "$MAX_NEW" --batch_size "$BATCH_SIZE" \
      --shard_idx "$i" --num_shards "$NUM_SHARDS" &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] a shard failed"; exit 1; }
  cat "$OUT_DIR/${IN_STEM}_next.shard"*.jsonl > "$OUT_DIR/${IN_STEM}_next.jsonl"
fi

echo "[verify] $(wc -l < "$OUT_DIR/${IN_STEM}_next.jsonl") rows"
echo "[$(date)] gen_next_7b ($IN_STEM) done"
