#!/bin/bash
#SBATCH --job-name=prm800k_mtml_encode
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Encode the PRM800K held-out test set at MULTIPLE layers x first/last step token
# in ONE forward pass, sharded across the 4 H100s, then merge. Output is a 4D
# array (n, L, T=2, H) so all layer/token ablation is a local re-slice.
#
# Build prm800k_heldout_test.jsonl first with build_prm800k_heldout_test.py.
#
# Usage:
#   sbatch slurm/encode_prm800k_multitoken_test_tamia.sh
#   STEM=prm800k_heldout_test LAYERS="11 17 20 22 25 28" FORCE=1 sbatch ...

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
DATA_DIR="${DATA_DIR:-$SCRATCH/cot_mech/prestudy_v1/data}"
OUT="${OUT:-$HOME/CoT-checker/runs/s1_model_size_dense/qwen2_5_7b/prm_multitoken}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
STEM="${STEM:-prm800k_heldout_test}"
LAYERS="${LAYERS:-11 17 20 22 25 28}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_SHARDS="${NUM_SHARDS:-4}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"

mkdir -p "$OUT" "$HF_CACHE"
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-prm800k_mtml_encode}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
git_commit : $GIT_COMMIT
model      : $MODEL_NAME_OR_PATH
split      : ${STEM}.jsonl   layers: $LAYERS   shards: $NUM_SHARDS
out        : $OUT
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_FLAG+=(--force); fi

# one shard per GPU, in parallel
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_DIR=$(printf "%s/shard_%02d" "$OUT" "$i")
  CUDA_VISIBLE_DEVICES=$i python scripts/encode_prm800k_multitoken_multilayer.py \
    --data_dir "$DATA_DIR" --out_dir "$SHARD_DIR" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --splits "${STEM}.jsonl:${STEM}" \
    --layers $LAYERS --max_seq_len -1 \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" \
    --batch_size "$BATCH_SIZE" --model_dtype float16 --save_dtype float16 \
    "${FORCE_FLAG[@]}" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "[$(date -Iseconds)] all $NUM_SHARDS shards done; merging"

python scripts/merge_prm800k_multitoken_shards.py \
  --shard_root "$OUT" --stem "$STEM" --out_dir "$OUT" "${FORCE_FLAG[@]}"

echo "[$(date)] mtml encode complete -> $OUT/${STEM}_{h,y,meta}"
