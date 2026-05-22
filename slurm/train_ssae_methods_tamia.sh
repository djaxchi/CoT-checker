#!/bin/bash
# SSAE phase-1 production pipeline on TamIA.
#
# Waves:
#   Wave 0:  small smoke (functional check, ssae_mixed, single GPU, bs=1)
#   Wave 0b: production-memory smoke (DDP, bs=8, accum=16, 1 iter) -- catches OOM
#            before launching the three full methods.
#   Wave 1:  ssae_positive    (DDP, 4x H100)
#   Wave 2:  ssae_mixed       (DDP, 4x H100)
#   Wave 3:  ssae_contrastive (DDP, 4x H100)
#   Wave 4:  merge leaderboard
#
# Each SSAE training uses all 4 GPUs (DDP). Methods run sequentially because
# each one needs the full node; running them in parallel would oversubscribe.
#
# Memory note: a previous run OOMed at bs=16 x accum=8 on 4x H100 80GB
# (peak ~79 GB). The defaults below use bs=8 x accum=16, preserving the
# effective global batch of 128 (8 x 16 x 1) while halving activation
# memory. If this still OOMs, the approved next fallback is bs=4 x accum=32.

#SBATCH --job-name=ssae_methods
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Debug walltime for the max_iters=30 sweep. For real/full SSAE training,
# increase this walltime.
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Pre-flight: run `bash scripts/check_ssae_deps.sh` once before submitting
# this job to confirm `pip install --no-index transformers tokenizers` works
# on the TamIA offline wheelhouse. The job below assumes that check passed.

module load StdEnv/2023 python/3.12

PROJECT_ROOT="$HOME/CoT-checker"
RUN_ROOT="$SCRATCH/cot_mech/prestudy_v1"
DATA_DIR="$RUN_ROOT/data"
OUT_ROOT="$RUN_ROOT/runs"
SMOKE_OUT="$OUT_ROOT/ssae_smoke"
MEM_SMOKE_OUT="$OUT_ROOT/ssae_mem_smoke"

mkdir -p "$OUT_ROOT"

# Offline: nothing in compute jobs may hit the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Reduces fragmentation under the tight 80 GB H100 budget.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy tqdm pyyaml transformers

MODEL_PATH="Qwen/Qwen2.5-1.5B"
MAX_ITERS="${MAX_ITERS:-30}"
# Post-OOM defaults: per-GPU bs=8, accum=16 -> effective global batch = 128.
# Approved next fallback if this still OOMs: BATCH_SIZE=4 GRAD_ACCUM=32.
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LEARNING_RATE:-1e-6}"

run_method () {
  local method="$1"
  local nproc="$2"
  shift 2
  local out_dir="$OUT_ROOT/$method"
  echo "[$(date)] === BEGIN $method (nproc=$nproc, bs=$BATCH_SIZE accum=$GRAD_ACCUM) ==="
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

# ---- Wave 0: functional smoke test (single GPU) ---------------------------
echo "[$(date)] Wave 0: functional smoke test"
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
echo "[$(date)] Wave 0 functional smoke complete"

# ---- Wave 0b: production-memory smoke (DDP, same bs/accum as production) --
# Runs the actual production memory configuration (4-GPU DDP, bs=8, accum=16)
# for max_iters=1 so we catch any OOM here instead of after a 30-iter wait.
# `set -e` will abort the job if this step fails.
echo "[$(date)] Wave 0b: production-memory smoke (DDP, bs=$BATCH_SIZE accum=$GRAD_ACCUM)"
python scripts/run_ssae_method.py \
  --method ssae_mixed \
  --data_dir "$DATA_DIR" \
  --out_dir "$MEM_SMOKE_OUT" \
  --model_name_or_path "$MODEL_PATH" \
  --local_files_only \
  --phase 1 \
  --sparsity_factor 1 \
  --l1_weight 1e-4 \
  --bce_weight 0.1 \
  --max_seq_len 2048 \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --learning_rate 1e-6 \
  --min_lr 1e-7 \
  --warmup_iters 0 \
  --max_iters 1 \
  --nproc_per_node 4 \
  --skip_extract \
  --skip_probe \
  --seed 42
echo "[$(date)] Wave 0b production-memory smoke complete"

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
