#!/bin/bash
#SBATCH --job-name=s4_forks
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# S4 matched correct/incorrect forks: encode 4 prefixes per fork (p0, prefix,
# prefix+correct, prefix+wrong) at L20/L28 with Qwen2.5-7B, 4 GPU shards + merge.
# Requires runs/contrib_cluster/forks.jsonl (built locally by
# scripts/analysis/s4_contrib_forks.py and committed/pulled or rsynced here).
#
# Usage:
#   sbatch slurm/s4_contrib_forks_tamia.sh
#   FORCE=1 LIMIT=100 sbatch --time=00:15:00 slurm/s4_contrib_forks_tamia.sh  # smoke

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/contrib_cluster}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYERS="${LAYERS:-20 28}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-}"
FORCE="${FORCE:-}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
test -f "$RUN_DIR/forks.jsonl" || { echo "missing $RUN_DIR/forks.jsonl"; exit 1; }

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy pandas pyarrow

FORCE_FLAG=""; [ -n "$FORCE" ] && FORCE_FLAG="--force"
LIMIT_FLAG=""; [ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-s4_forks}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH   layers: $LAYERS
forks      : $RUN_DIR/forks.jsonl ($(wc -l < "$RUN_DIR/forks.jsonl") forks)
================================================================
BANNER

pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/s4_contrib_extract_forks.py \
    --forks "$RUN_DIR/forks.jsonl" \
    --out_dir "$RUN_DIR/forks_hidden" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --layers $LAYERS --batch_size "$BATCH_SIZE" --max_seq_len "$MAX_SEQ_LEN" \
    --shard_idx $i --num_shards 4 --seed 42 $FORCE_FLAG $LIMIT_FLAG \
    > "$RUN_DIR/forks_extract_shard$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/s4_contrib_extract_forks.py --merge --num_shards 4 \
  --out_dir "$RUN_DIR/forks_hidden" --layers $LAYERS

echo "[done] $(date -Iseconds) -> $RUN_DIR/forks_hidden"
