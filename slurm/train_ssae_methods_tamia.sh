#!/bin/bash
# SSAE phase-1 production pipeline on TamIA.
#
# Waves:
#   Wave 0: smoke test (ssae_mixed, --nproc_per_node=1)
#   Wave 1: ssae_positive    (DDP, 4x H100)
#   Wave 2: ssae_mixed       (DDP, 4x H100)
#   Wave 3: ssae_contrastive (DDP, 4x H100)
#   Wave 4: merge leaderboard
#
# Each SSAE training uses all 4 GPUs (DDP). Methods run sequentially because
# each one needs the full node; running them in parallel would oversubscribe.

#SBATCH --job-name=ssae_methods
#SBATCH --account=aip-${PI_NAME}
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Pre-flight: run `bash scripts/check_ssae_deps.sh` once before submitting
# this job to confirm `pip install --no-index transformers tokenizers` works
# on the TamIA offline wheelhouse. The job below assumes that check passed.

module load StdEnv/2023 python/3.12

PROJECT_ROOT="$HOME/Code/CoT-checker"
RUN_ROOT="$SCRATCH/cot_mech/prestudy_v1"
DATA_DIR="$RUN_ROOT/data"
OUT_ROOT="$RUN_ROOT/runs"
SMOKE_OUT="$OUT_ROOT/ssae_smoke"

mkdir -p "$OUT_ROOT"

# Offline: nothing in compute jobs may hit the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy tqdm pyyaml transformers

MODEL_PATH="Qwen/Qwen2.5-1.5B"
MAX_ITERS="${MAX_ITERS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LEARNING_RATE:-1e-6}"

run_method () {
  local method="$1"
  local nproc="$2"
  shift 2
  local out_dir="$OUT_ROOT/$method"
  echo "[$(date)] === BEGIN $method (nproc=$nproc) ==="
  python scripts/run_ssae_method.py \
    --method "$method" \
    --data_dir "$DATA_DIR" \
    --out_dir "$out_dir" \
    --model_name_or_path "$MODEL_PATH" \
    --local_files_only \
    --phase 1 \
    --sparsity_factor 1 \
    --l1_weight 1e-4 \
    --bce_weight 0.1 \
    --max_seq_len 2048 \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --min_lr 1e-7 \
    --warmup_iters 2 \
    --max_iters "$MAX_ITERS" \
    --nproc_per_node "$nproc" \
    --seed 42 \
    "$@"
  echo "[$(date)] === END   $method ==="
}

# ---- Wave 0: smoke test ----------------------------------------------------
echo "[$(date)] Wave 0: smoke test"
python scripts/run_ssae_method.py \
  --method ssae_mixed \
  --data_dir "$DATA_DIR" \
  --out_dir "$SMOKE_OUT" \
  --model_name_or_path "$MODEL_PATH" \
  --local_files_only \
  --phase 1 \
  --sparsity_factor 1 \
  --l1_weight 1e-4 \
  --bce_weight 0.1 \
  --max_seq_len 2048 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --learning_rate 1e-6 \
  --min_lr 1e-7 \
  --warmup_iters 0 \
  --max_iters 2 \
  --nproc_per_node 1 \
  --extract_batch_size 2 \
  --epochs_probe 2 \
  --probe_batch_size 32 \
  --smoke \
  --smoke_train_n 128 \
  --smoke_val_n 32 \
  --seed 42
echo "[$(date)] Wave 0 smoke complete"

# ---- Wave 1-3: SSAE methods, DDP over all 4 H100 GPUs ----------------------
run_method ssae_positive    4
run_method ssae_mixed       4
run_method ssae_contrastive 4

# ---- Wave 4: leaderboard ---------------------------------------------------
python scripts/merge_ssae_leaderboard.py \
  --runs_dir "$OUT_ROOT" \
  --out_csv "$OUT_ROOT/leaderboard_ssae.csv" \
  --out_md "$OUT_ROOT/leaderboard_ssae.md"

echo "[$(date)] All SSAE methods done"
