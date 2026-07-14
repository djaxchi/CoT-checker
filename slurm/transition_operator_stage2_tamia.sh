#!/bin/bash
#SBATCH --job-name=to_stage2
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:30:00
#SBATCH --output=%x-%j.out

# transition_operator_v0 Stage 2 (v0.3): training-array extraction at L20 (4 GPU
# shards + merge), transform prep (train-split PCA-64 of dL + residualization),
# then the three-arm ablation A / B / AB x seeds {0,1,2}, 9 runs round-robin over
# 4 GPUs. Requires forks.jsonl, splits.json, stage1/step_labels.parquet.
#
# Usage:  sbatch slurm/transition_operator_stage2_tamia.sh
#         SKIP_EXTRACT=1 sbatch ...   # arrays already extracted

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/transition_operator}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYER="${LAYER:-20}"
EPOCHS="${EPOCHS:-30}"
SKIP_EXTRACT="${SKIP_EXTRACT:-}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
for f in "$RUN_DIR/forks.jsonl" "$RUN_DIR/splits.json" \
         "$RUN_DIR/stage1/step_labels.parquet"; do
  test -e "$f" || { echo "missing $f"; exit 1; }
done

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn pandas pyarrow

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-to_stage2}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH   layer: $LAYER   epochs: $EPOCHS
forks      : $(wc -l < "$RUN_DIR/forks.jsonl")
================================================================
BANNER

mkdir -p "$RUN_DIR/stage2/arrays"
if [ -z "$SKIP_EXTRACT" ]; then
  pids=()
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/transition_operator/to_extract_train.py \
      --run_dir "$RUN_DIR" --model_name_or_path "$MODEL_NAME_OR_PATH" \
      --local_files_only --layer "$LAYER" --shard_idx $i --num_shards 4 \
      --device cuda > "$RUN_DIR/stage2/extract_shard$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  python scripts/transition_operator/to_extract_train.py --run_dir "$RUN_DIR" --merge
fi

python scripts/transition_operator/to_train.py --run_dir "$RUN_DIR" --prep_only \
  --layer "$LAYER"

RUNS=(A:0 B:0 AB:0 A:1 B:1 AB:1 A:2 B:2 AB:2)
pids=()
for g in 0 1 2 3; do
  (
    i=0
    for r in "${RUNS[@]}"; do
      if [ $(( i % 4 )) -eq $g ]; then
        arm="${r%%:*}"; seed="${r##*:}"
        echo "[gpu$g] $arm seed $seed $(date -Iseconds)"
        CUDA_VISIBLE_DEVICES=$g python scripts/transition_operator/to_train.py \
          --run_dir "$RUN_DIR" --arm "$arm" --seed "$seed" --layer "$LAYER" \
          --epochs "$EPOCHS" --local_files_only --device cuda \
          --model_name_or_path "$MODEL_NAME_OR_PATH" \
          > "$RUN_DIR/stage2/train_${arm}_s${seed}.log" 2>&1
      fi
      i=$(( i + 1 ))
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "[done] $(date -Iseconds) -> $RUN_DIR/stage2"
grep -h '"best_val_total"' "$RUN_DIR"/stage2/*_seed*/metrics.json 2>/dev/null || true
