#!/bin/bash
# Smoke test for SSAE phase-1 (single GPU). Use BEFORE the full SLURM job.
#
# Verifies:
#   1. Model initializes (3x Qwen2.5-1.5B + autoencoder + projection_mlp).
#   2. Forward pass works with the no-truncation tokenizer pipeline.
#   3. CE alignment matches input_ids (assertion inside compute_loss).
#   4. Checkpoint saves and reloads.
#   5. Latents extract with expected shape.
#   6. Linear probe trains.
#   7. ProcessBench evaluator runs.
#   8. No dense .npy hidden-state cache is touched (latents come from text JSONL).
#
# Reads:
#   $SCRATCH/cot_mech/prestudy_v1/data/{prm800k_pos_base_20k,prm800k_mixed_train_40k,
#                                       prm800k_probe_train_40k,prm800k_val_1k,
#                                       processbench_gsm8k}.jsonl
# Writes:
#   $SCRATCH/cot_mech/prestudy_v1/runs/ssae_smoke/

set -euo pipefail

RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/prestudy_v1}"
DATA_DIR="$RUN_ROOT/data"
OUT_DIR="$RUN_ROOT/runs/ssae_smoke"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR"

cd "$(dirname "$0")/.."

python scripts/run_ssae_method.py \
  --method ssae_mixed \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
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

echo "[SMOKE OK] outputs in $OUT_DIR"
