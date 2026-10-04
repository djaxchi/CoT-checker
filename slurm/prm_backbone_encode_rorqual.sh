#!/bin/bash
#SBATCH --job-name=prm_backbone_encode
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# prm_backbone_v1: encode step spans with Qwen2.5-Math-PRM-7B instead of
# Qwen3-8B, on one GPU, for the per-GPU Alliance clusters (Rorqual).
#
# Everything but the backbone is the Instruct arm's protocol: same frozen
# splits, same verifier template, span-only store, bfloat16 forward pass.
#   WHAT=prm  one shard (SHARD of NUM_SHARDS) of PRM800K val, test, train
#   WHAT=pb   all ProcessBench subsets, every shard in turn (they are small)
#
# Loading: the checkpoint's architecture is Qwen2ForProcessRewardModel, but its
# model_type is qwen2 and it ships lm_head.weight, so AutoModelForCausalLM builds
# Qwen2ForCausalLM with zero missing keys (checked on the config, 2026-10-02);
# only the reward head score.* is dropped. No remote code, any transformers.
#
# Layer: LAYER=27 is resid_post of block 26 of 28, the second-to-last block,
# matching the Instruct arm's LAYER=35 (block 34 of 36).
#
# The PRM tokenizer and Qwen3-8B's tokenise all 2,000 test rows identically
# under the verifier template, so both backbones read the same token sequence.
# NO INTERNET on compute nodes: weights must be in $HF_HOME.

set -euo pipefail
: "${WHAT:?prm or pb}" "${RUN_ROOT:?}"
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-PRM-7B}"
LAYER="${LAYER:-27}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/logs"
echo "git $(git rev-parse --short HEAD) what=$WHAT shard=${SHARD:-all} model=$MODEL layer=$LAYER"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy 2>&1 | tail -1

common=(--model_name_or_path "$MODEL" --local_files_only --span_only
        --layer "$LAYER" --max_seq_len 2048 --batch_size "$BATCH_SIZE"
        --model_dtype bfloat16 --num_shards "$NUM_SHARDS")

if [[ "$WHAT" == prm ]]; then
  : "${SHARD:?}"
  python scripts/encode_prm800k_token_store.py \
    --data_dir "$RUN_ROOT/data" --rep_root "$RUN_ROOT/repstore/step_spans" \
    --splits prm800k_val_5k.jsonl:val_5k prm800k_test_2k.jsonl:test_2k \
             prm800k_probe_train_full.jsonl:probe_train_full \
    --shard_idx "$SHARD" "${common[@]}"
else
  specs=()
  for s in gsm8k math olympiadbench omnimath; do
    specs+=("$s:$RUN_ROOT/processbench/processbench_$s.jsonl")
  done
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    python scripts/encode_processbench_token_store.py --raw_specs "${specs[@]}" \
      --rep_root "$RUN_ROOT/repstore/pb_step_spans" --shard_idx "$i" "${common[@]}"
  done
fi
du -sh "$RUN_ROOT"/repstore/* 2>/dev/null || true
echo "[$(date)] done"
