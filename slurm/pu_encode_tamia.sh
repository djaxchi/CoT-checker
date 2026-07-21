#!/bin/bash
#SBATCH --job-name=pu_encode
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --output=%x-%j.out

# progress_usefulness_v0 P2 encode: last/mean/max pooling over the step span at
# layers 12 20 26 28, for the progress/neutral items. ~3.9k items encode fast, so
# a single GPU / single shard (no merge). The sweep (CPU, local) filters to the
# causally-confirmed forks from P1 at analysis time.
#
# Usage:  sbatch slurm/pu_encode_tamia.sh
#         LAYERS="12 20 26 28" sbatch ...

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
DATA_ROOT="${DATA_ROOT:-/scratch/d/dchikhi/cot_mech/progress_usefulness_v0}"
PAIRS_DIR="${PAIRS_DIR:-$DATA_ROOT/data}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/enc}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYERS="${LAYERS:-12 20 26 28}"
BATCH_SIZE="${BATCH_SIZE:-16}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
for f in pu_train_items.jsonl pu_val_items.jsonl; do
  test -e "$PAIRS_DIR/$f" || { echo "missing $PAIRS_DIR/$f (run pu_build_pairs first)"; exit 1; }
done
mkdir -p "$OUT_DIR"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy pandas pyarrow

cat <<BANNER
================================================================
job     : ${SLURM_JOB_NAME:-pu_encode}  id ${SLURM_JOB_ID:-N/A}
host    : $(hostname)   date $(date -Iseconds)
out_dir : $OUT_DIR   layers $LAYERS
================================================================
BANNER

CUDA_VISIBLE_DEVICES=0 python scripts/progress_usefulness/pu_encode.py \
  --data_dir "$PAIRS_DIR" --out_dir "$OUT_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --splits pu_train_items.jsonl:pu_train pu_val_items.jsonl:pu_val \
  --layers $LAYERS --shard_idx 0 --num_shards 1 --batch_size "$BATCH_SIZE" --force

echo "[pu_encode] done $(date -Iseconds)"
