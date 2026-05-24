#!/bin/bash
# Diagnostic: does the contrastive auxiliary BCE actually learn anything
# when its loss weight is raised from 0.1 -> 1.0?
#
# Context (from the previous full ssae_contrastive run):
#   bce_weight=0.1, max_iters=30
#   aux_bce_start = 0.6931
#   aux_bce_end   = 0.6927   (delta ~= -0.0004, essentially random)
#
# Labels are balanced (20k / 20k in prm800k_mixed_train_40k, 500 / 500 in
# prm800k_val_1k), so the flat aux_bce is NOT a class-imbalance artifact.
#
# This run isolates ONE variable: bce_weight 0.1 -> 1.0. Everything else
# is the known-stable Phase 1 recipe (eager attention, gradient
# checkpointing, latent_norm_eps=1e-8, train_attn_mask_ratio=0.1, bs=4 x
# accum=32 x 4 GPUs -> effective global batch = 512).
#
# Stop criterion (read aux_bce from the *.out file while it runs):
#   * aux_bce ~ 0.692-0.693 after 10-15 iters  -> CANCEL the job. Raising
#     bce_weight did not help; the issue is wiring / gradient flow /
#     representation, not loss weight.
#   * aux_bce drops below 0.69                  -> let it finish 30 iters.
#   * aux_bce reaches 0.65-0.67                 -> contrastive signal IS
#     learning; after this run, launch extraction + probe eval for this
#     checkpoint (a separate job; this script skips both stages).
#
# This script does NOT touch the production
# slurm/train_ssae_methods_tamia.sh recipe. It does NOT run the full
# leaderboard. It does NOT run ssae_positive or ssae_mixed.

#SBATCH --job-name=ssae_ctr_bce1_diag
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module load StdEnv/2023 python/3.12

PROJECT_ROOT="$HOME/CoT-checker"
RUN_ROOT="$SCRATCH/cot_mech/prestudy_v1"
DATA_DIR="$RUN_ROOT/data"
OUT_DIR="$RUN_ROOT/runs/ssae_contrastive_bce1_diag"

TRAIN_JSONL="$DATA_DIR/prm800k_mixed_train_40k.jsonl"
VAL_JSONL="$DATA_DIR/prm800k_val_1k.jsonl"

# Refuse to clobber a previous diagnostic run.
if [[ -e "$OUT_DIR" ]]; then
  echo "[ERR] $OUT_DIR already exists; refusing to overwrite. Move/rename it and resubmit." >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

# Offline: compute nodes have no network. Match the production recipe.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HF cache locations (only set if the submitter has not already exported them).
: "${HF_HOME:=/scratch/d/dchikhi/hf_cache}"
: "${HF_HUB_CACHE:=/scratch/d/dchikhi/hf_cache/hub}"
: "${TRANSFORMERS_CACHE:=/scratch/d/dchikhi/hf_cache/hub}"
export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE

cd "$PROJECT_ROOT"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy tqdm pyyaml transformers

MODEL_PATH="Qwen/Qwen2.5-1.5B"
METHOD="ssae_contrastive"
BCE_WEIGHT="1.0"
MAX_ITERS=30
BATCH_SIZE=4
GRAD_ACCUM=32
NPROC=4
ATTN_IMPL="eager"
TRAIN_ATTN_MASK_RATIO="0.1"
LATENT_NORM_EPS="1e-8"
CE_CHUNK_SIZE=2048
GIT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
EFFECTIVE_GLOBAL_BATCH=$(( BATCH_SIZE * GRAD_ACCUM * NPROC ))

echo "========================================================================"
echo " SSAE contrastive bce_weight=1.0 diagnostic"
echo "------------------------------------------------------------------------"
echo " git_commit             : $GIT_COMMIT"
echo " method                 : $METHOD"
echo " out_dir                : $OUT_DIR"
echo " bce_weight             : $BCE_WEIGHT"
echo " max_iters              : $MAX_ITERS"
echo " attn_implementation    : $ATTN_IMPL"
echo " gradient_checkpointing : ON"
echo " train_attn_mask_ratio  : $TRAIN_ATTN_MASK_RATIO"
echo " latent_norm_eps        : $LATENT_NORM_EPS"
echo " ce_chunk_size          : $CE_CHUNK_SIZE"
echo " batch_size (per GPU)   : $BATCH_SIZE"
echo " grad_accum_steps       : $GRAD_ACCUM"
echo " nproc_per_node         : $NPROC"
echo " effective global batch : $EFFECTIVE_GLOBAL_BATCH  (bs * accum * nproc)"
echo " model                  : $MODEL_PATH"
echo " train_jsonl            : $TRAIN_JSONL"
echo " val_jsonl              : $VAL_JSONL"
echo "========================================================================"

# Use the existing run_ssae_method.py entry point with --skip_extract /
# --skip_probe so this diagnostic ONLY does SSAE training. Extraction and
# probe evaluation are deferred until we confirm aux_bce actually moves.
python scripts/run_ssae_method.py \
  --method "$METHOD" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --model_name_or_path "$MODEL_PATH" \
  --local_files_only \
  --phase 1 \
  --sparsity_factor 1 \
  --n_inputs 1536 \
  --l1_weight 1e-4 \
  --bce_weight "$BCE_WEIGHT" \
  --max_seq_len 2048 \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --learning_rate 1e-6 \
  --min_lr 1e-7 \
  --warmup_iters 2 \
  --max_iters "$MAX_ITERS" \
  --nproc_per_node "$NPROC" \
  --ce_chunk_size "$CE_CHUNK_SIZE" \
  --train_attn_mask_ratio "$TRAIN_ATTN_MASK_RATIO" \
  --attn_implementation "$ATTN_IMPL" \
  --latent_norm_eps "$LATENT_NORM_EPS" \
  --max_grad_norm 1.0 \
  --gradient_checkpointing \
  --skip_extract \
  --skip_probe \
  --seed 42

echo "[$(date)] diagnostic done; checkpoints in $OUT_DIR"

# -----------------------------------------------------------------------------
# Monitoring helper (run from the login node while the job is alive):
#
#   grep -nE "train_aux_bce|train_nll|val_recon_ce|Saved final|NonFiniteError|OutOfMemory|traceback|error" \
#        ssae_ctr_bce1_diag-<JOBID>.out | tail -100
#
# Decision points (see header):
#   aux_bce ~0.693 after 10-15 iters -> scancel; wiring issue, not loss weight.
#   aux_bce < 0.69                    -> let it finish.
#   aux_bce ~0.65-0.67                -> launch extraction + probe for this run.
# -----------------------------------------------------------------------------
