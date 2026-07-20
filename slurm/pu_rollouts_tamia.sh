#!/bin/bash
#SBATCH --job-name=pu_rollouts
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# progress_usefulness_v0 P1: causal-label gate via free-generation rollouts.
# Pilot (PILOT_PAIRS pairs on GPU 0) validates the pipeline and prints an early
# gate + s/pair, then the full 4-shard in-node run over all +1/0 pairs, then the
# CPU merge that writes pu_gates.json and pu_confirmed_forks.jsonl.
#
# Usage:  sbatch slurm/pu_rollouts_tamia.sh
#         K_ROLLOUTS=16 sbatch ...          # more rollouts per context
#         PILOT_PAIRS=40 sbatch ...

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
DATA_ROOT="${DATA_ROOT:-/scratch/d/dchikhi/cot_mech/progress_usefulness_v0}"
PAIRS_DIR="${PAIRS_DIR:-$DATA_ROOT/data}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/rollouts}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
K_ROLLOUTS="${K_ROLLOUTS:-8}"
PILOT_PAIRS="${PILOT_PAIRS:-24}"

source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
for f in pu_train_pairs.jsonl pu_val_pairs.jsonl pu_train_items.jsonl pu_val_items.jsonl; do
  test -e "$PAIRS_DIR/$f" || { echo "missing $PAIRS_DIR/$f (run pu_build_pairs first)"; exit 1; }
done
mkdir -p "$OUT_DIR"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scipy scikit-learn pandas pyarrow

cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-pu_rollouts}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
pairs_dir  : $PAIRS_DIR
out_dir    : $OUT_DIR   k_rollouts $K_ROLLOUTS   pilot $PILOT_PAIRS
================================================================
BANNER

# ---- pilot: sanity + timing + early gate on a small slice --------------------
CUDA_VISIBLE_DEVICES=0 python scripts/progress_usefulness/pu_rollouts.py \
  --pairs_dir "$PAIRS_DIR" --out_dir "$OUT_DIR/pilot" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --k_rollouts "$K_ROLLOUTS" --limit "$PILOT_PAIRS" --shard_id 0 --num_shards 1
python scripts/progress_usefulness/pu_rollouts.py --out_dir "$OUT_DIR/pilot" --merge
echo "[pu_rollouts] pilot done; gate above is from $PILOT_PAIRS pairs"

# ---- full run, 4 in-node shards ----------------------------------------------
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python scripts/progress_usefulness/pu_rollouts.py \
    --pairs_dir "$PAIRS_DIR" --out_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --k_rollouts "$K_ROLLOUTS" --shard_id $g --num_shards 4 \
    > "$OUT_DIR/pu_rollouts_shard$g-${SLURM_JOB_ID:-manual}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# ---- merge + gate (CPU) ------------------------------------------------------
python scripts/progress_usefulness/pu_rollouts.py --out_dir "$OUT_DIR" --merge
echo "[pu_rollouts] done $(date -Iseconds)"
