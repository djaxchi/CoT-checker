#!/bin/bash
#SBATCH --job-name=s4_contrib
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# S4 contrib-cluster pipeline (exploratory, unsupervised — no correctness labels):
#   1. audit + sample 3000 golden-path PRM800K trajectories (CPU)
#   2. hidden-state extraction over cumulative prefixes, 4 GPU shards + merge
#   3. step representations (state/qres/contribution) at L20 + L28
#   4. weak regex tags + surface features
#   5. PCA-50 -> HDBSCAN clustering + enrichment + cluster cards + UMAP plots
#   6. combined summary CSVs + report.md skeleton
#
# Usage:
#   sbatch slurm/s4_contrib_cluster_tamia.sh
#   FORCE=1 LIMIT=50 sbatch --time=00:20:00 slurm/s4_contrib_cluster_tamia.sh  # smoke

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RAW_PRM800K_DIR="${RAW_PRM800K_DIR:-$SCRATCH/cot_mech/raw/prm800k}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/contrib_cluster}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
LAYERS="${LAYERS:-20 28}"
N_TRAJ="${N_TRAJ:-3000}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"   # overlength trajectories get trailing steps
                                     # trimmed to fit, never silently dropped
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-}"          # smoke: cap trajectories at extraction
FORCE="${FORCE:-}"

# 7B weights live in the S1 HF cache (models.env HF_CACHE_ROOT), not $SCRATCH/hf_cache.
source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR"
FORCE_FLAG=""; [ -n "$FORCE" ] && FORCE_FLAG="--force"
LIMIT_FLAG=""; [ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-s4_contrib}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH   layers: $LAYERS   max_seq_len: $MAX_SEQ_LEN
raw        : $RAW_PRM800K_DIR
run_dir    : $RUN_DIR
n_traj     : $N_TRAJ (max_steps $MAX_STEPS)  limit: ${LIMIT:-none}
================================================================
BANNER

echo "[1/6] audit + sample trajectories"
python scripts/s4_contrib_audit.py \
  --raw_dir "$RAW_PRM800K_DIR" --out_dir "$RUN_DIR" \
  --n_trajectories "$N_TRAJ" --max_steps "$MAX_STEPS" \
  --tokenizer_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --max_seq_len "$MAX_SEQ_LEN" --seed 42 $FORCE_FLAG

echo "[2/6] hidden-state extraction (4 GPU shards)"
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/s4_contrib_extract.py \
    --trajectories "$RUN_DIR/trajectories.jsonl" \
    --out_dir "$RUN_DIR/hidden_states" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --layers $LAYERS --batch_size "$BATCH_SIZE" --max_seq_len "$MAX_SEQ_LEN" \
    --shard_idx $i --num_shards 4 --seed 42 $FORCE_FLAG $LIMIT_FLAG \
    > "$RUN_DIR/extract_shard$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/s4_contrib_extract.py --merge --num_shards 4 \
  --out_dir "$RUN_DIR/hidden_states" --layers $LAYERS

echo "[3/6] step representations"
python scripts/analysis/s4_contrib_reprs.py \
  --hidden_dir "$RUN_DIR/hidden_states" --out_dir "$RUN_DIR/reprs" \
  --layers $LAYERS $FORCE_FLAG

echo "[4/6] regex tags + surface features"
python scripts/analysis/s4_contrib_tags.py --run_dir "$RUN_DIR" $FORCE_FLAG

echo "[5/6] clustering + enrichment + cards + UMAP"
python scripts/analysis/s4_contrib_cluster.py \
  --run_dir "$RUN_DIR" --layers $LAYERS --seed 42

echo "[6/6] combined summary + report skeleton"
python scripts/analysis/s4_contrib_summary.py --run_dir "$RUN_DIR"

echo "[done] $(date -Iseconds) -> $RUN_DIR"
