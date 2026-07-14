#!/bin/bash
#SBATCH --job-name=to_stage1
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# transition_operator_v0 Stage 1: baseline kill gate.
# 1) operation labels + coverage (CPU), 2) baseline reps extraction at L20/L28
# (4 GPU shards + merge), 3) decodability + cross-problem retrieval eval with the
# kill gate. Requires runs/transition_operator/forks.jsonl (built on login node).
#
# Usage:  sbatch slurm/transition_operator_stage1_tamia.sh

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/transition_operator}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYERS="${LAYERS:-20 28}"
D_Z="${D_Z:-64}"

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
pip install --no-index torch transformers numpy scipy scikit-learn pandas pyarrow

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-to_stage1}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH   layers: $LAYERS   d_z: $D_Z
forks      : $RUN_DIR/forks.jsonl ($(wc -l < "$RUN_DIR/forks.jsonl") forks)
================================================================
BANNER

# 1) labels + coverage (CPU)
python scripts/transition_operator/to_labels.py --run_dir "$RUN_DIR"

# 2) baseline reps extraction, 4 GPU shards
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/transition_operator/to_extract_baselines.py \
    --forks "$RUN_DIR/forks.jsonl" --out_dir "$RUN_DIR/stage1" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --layers $LAYERS --shard_idx $i --num_shards 4 --device cuda \
    > "$RUN_DIR/stage1/extract_shard$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/transition_operator/to_extract_baselines.py \
  --out_dir "$RUN_DIR/stage1" --layers $LAYERS --merge

# 3) baseline eval + kill gate, per layer
for L in $LAYERS; do
  python scripts/transition_operator/to_stage1.py \
    --run_dir "$RUN_DIR" --layer "$L" --d_z "$D_Z"
done

echo "[done] $(date -Iseconds) -> $RUN_DIR/stage1"
