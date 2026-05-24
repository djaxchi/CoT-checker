#!/bin/bash
# Diagnostic: does the SSAE-contrastive aux_bce move when the aux_head gets
# its own probe-style LR (1e-3) instead of the trunk LR (1e-6)?
#
# Single variable changed vs. the production recipe:
#   --aux_learning_rate 1e-3   (was: implicit, = learning_rate = 1e-6)
#
# bce_weight is kept at 0.1 on purpose. The hypothesis is "head LR too low",
# not "loss weight too low". Changing both at once would confound the test.
#
# Short run (max_iters=10) to fail fast. Decision table after the run:
#
#   aux_bce < 0.69 by iter 5-10                  -> head LR was the issue.
#   aux_logit_std grows from ~0 to >> 0           -> head is actually moving.
#   aux_head_w_norm changes vs. init              -> optimizer is updating it.
#   aux_head_grad_norm > 0                        -> BCE gradient reaches head.
#   label_mean ~ 0.5                              -> labels are balanced.
#
# If aux_bce stays ~0.693 AND aux_logit_std grows AND aux_head_w_norm moves,
# the issue is NOT the LR -- the labels are not linearly learnable from the
# unit-norm latents (representation problem, not optimization problem).
#
# This diagnostic SKIPS extraction and probe. Promote to a 30-iter run +
# extraction + probe only if the success criteria above are met.

#SBATCH --job-name=ssae_ctr_auxlr_diag
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module load StdEnv/2023 python/3.12

PROJECT_ROOT="$HOME/CoT-checker"
RUN_ROOT="$SCRATCH/cot_mech/prestudy_v1"
DATA_DIR="$RUN_ROOT/data"
OUT_DIR="$RUN_ROOT/runs/ssae_contrastive_auxlr1e-3_diag"

TRAIN_JSONL="$DATA_DIR/prm800k_mixed_train_40k.jsonl"
VAL_JSONL="$DATA_DIR/prm800k_val_1k.jsonl"

if [[ -e "$OUT_DIR" ]]; then
  echo "[ERR] $OUT_DIR already exists; refusing to overwrite. Move/rename and resubmit." >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
BCE_WEIGHT="0.1"
AUX_LR="1e-3"
TRUNK_LR="1e-6"
MAX_ITERS=10
BATCH_SIZE=4
GRAD_ACCUM=32
NPROC=4
GIT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
EFFECTIVE_GLOBAL_BATCH=$(( BATCH_SIZE * GRAD_ACCUM * NPROC ))

echo "========================================================================"
echo " SSAE contrastive aux_lr=$AUX_LR diagnostic (10 iters)"
echo "------------------------------------------------------------------------"
echo " git_commit             : $GIT_COMMIT"
echo " method                 : $METHOD"
echo " out_dir                : $OUT_DIR"
echo " bce_weight             : $BCE_WEIGHT   (unchanged vs. production)"
echo " learning_rate (trunk)  : $TRUNK_LR"
echo " aux_learning_rate      : $AUX_LR       (NEW: separate param group)"
echo " max_iters              : $MAX_ITERS    (short; promote to 30 if success)"
echo " attn_implementation    : eager"
echo " gradient_checkpointing : ON"
echo " train_attn_mask_ratio  : 0.1"
echo " latent_norm_eps        : 1e-8"
echo " ce_chunk_size          : 2048"
echo " batch_size (per GPU)   : $BATCH_SIZE"
echo " grad_accum_steps       : $GRAD_ACCUM"
echo " nproc_per_node         : $NPROC"
echo " effective global batch : $EFFECTIVE_GLOBAL_BATCH"
echo " model                  : $MODEL_PATH"
echo " train_jsonl            : $TRAIN_JSONL"
echo " val_jsonl              : $VAL_JSONL"
echo "========================================================================"

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
  --learning_rate "$TRUNK_LR" \
  --aux_learning_rate "$AUX_LR" \
  --min_lr 1e-7 \
  --warmup_iters 2 \
  --max_iters "$MAX_ITERS" \
  --nproc_per_node "$NPROC" \
  --ce_chunk_size 2048 \
  --train_attn_mask_ratio 0.1 \
  --attn_implementation eager \
  --latent_norm_eps 1e-8 \
  --max_grad_norm 1.0 \
  --gradient_checkpointing \
  --skip_extract \
  --skip_probe \
  --seed 42

echo "[$(date)] aux_lr diag done; logs in $OUT_DIR"

# -----------------------------------------------------------------------------
# Monitoring (login node):
#
#   grep -nE "aux-diag|aux-grad|train_aux_bce|trainable params|aux_head" \
#        ssae_ctr_auxlr_diag-<JOBID>.out | tail -120
#
# What to look at, in order:
#   1. "trainable params: base=... aux_head=1537 (base_lr=1.00e-06, aux_lr=1.00e-03)"
#   2. [aux-diag] aux_head_w_norm rising iter-over-iter -> optimizer works.
#   3. [aux-diag] logit_std rising from ~2e-2 -> head is producing signal.
#   4. [aux-grad] aux_head_grad_norm > 0 -> BCE gradients arrive.
#   5. train_aux_bce trending below 0.69 -> the head LR was the bottleneck.
#
# If 2-4 are healthy but 5 stays ~0.693, the labels are not linearly
# learnable from unit-norm latents -- that is a representation issue and
# the next step is P3 from the audit report (feed pre-normalize latents
# to the aux head).
# -----------------------------------------------------------------------------
