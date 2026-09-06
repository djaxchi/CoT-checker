#!/bin/bash
#SBATCH --job-name=reprobe_onpolicy
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

# Phases 7 and 8: train the ReProbe-style cell on Qwen's own trajectories with
# GPT-OSS step labels, then score it on the SAME held-out trajectories the frozen
# verifiers were scored on.
#
# The comparison this exists to make is narrow on purpose. The off-policy
# ReProbe cell (step_tokens x transformer, PRM800K-trained) already has scores on
# that evaluation pool. This trains the same architecture on the same backbone
# and evaluates on the same problems, so training distribution is the only thing
# that differs. Anything else varying would make the Phase 9 gate unreadable.
#
# Splits are by problem, never by trajectory: ten samples of one problem exist in
# this pool, and splitting by trajectory would let the probe memorise the problem
# and flatter its own validation curve.
#
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/reprobe_v1}"
ONPOLICY_ROOT="${ONPOLICY_ROOT:-$SCRATCH/cot_mech/onpolicy_v1}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
REP_ROOT="${REP_ROOT:-$RUN_ROOT/repstore/onpolicy_train_spans}"
EVAL_REP_ROOT="${EVAL_REP_ROOT:-$ONPOLICY_ROOT/repstore/onpolicy_step_spans}"
LAYER="${LAYER:-35}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEEDS="${SEEDS:-42 43 44}"
# The published-style cell and this project's own ReProbe approximation. Both
# read last-layer hidden states, which is already a deviation from the paper's
# attention-plus-logits features and is recorded in docs/reprobe_label_semantics.md.
CELLS="${CELLS:-step_tokens:transformer:d256,l2,f1024,h4 step_tokens:attn_query}"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$REP_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/cells"
LOG="$RUN_ROOT/logs/reprobe_onpolicy-${SLURM_JOB_ID:-$$}.log"

cd "$PROJECT_ROOT"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy 2>&1 | tail -1

echo "=== 1. splits (by problem, holdout re-verified) ==="
python scripts/onpolicy/build_onpolicy_splits.py \
  --traces "$RUN_ROOT/reprobe_train_judge_traces.jsonl" \
  --labels "$RUN_ROOT"/labels/labels.shard*.jsonl \
  --out_dir "$RUN_ROOT/splits" --stem reprobe_onpolicy \
  --holdout_outcomes "$ONPOLICY_ROOT/onpolicy_stage1_outcomes.jsonl" \
  --force 2>&1 | tee -a "$LOG"

echo
echo "=== 2. label audit (training stops if this fails) ==="
python scripts/analysis/onpolicy_label_audit.py \
  --labels "$RUN_ROOT"/labels/labels.shard*.jsonl \
  --traces "$RUN_ROOT/reprobe_train_judge_traces.jsonl" \
  --holdout_outcomes "$ONPOLICY_ROOT/onpolicy_stage1_outcomes.jsonl" \
  --out "$RUN_ROOT/label_audit.json" 2>&1 | tee -a "$LOG"

echo
echo "=== 3. encode train and val at layer $LAYER ==="
for split in train val; do
  f="$RUN_ROOT/splits/reprobe_onpolicy_${split}.jsonl"
  [[ -f "$f" ]] || { echo "[FATAL] missing $f" >&2; exit 2; }
  if [[ -d "$REP_ROOT/$split/shard_00" ]]; then
    echo "[skip] $split already encoded"; continue
  fi
  pids=()
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/encode_processbench_token_store.py \
      --raw_specs "${split}:${f}" --rep_root "$REP_ROOT" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
      --span_only --prompt_style verifier --layer "$LAYER" \
      --batch_size "$BATCH_SIZE" --model_dtype bfloat16 \
      --shard_idx "$i" --num_shards "$NUM_SHARDS" >>"$LOG" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] encoding $split failed" >&2; tail -30 "$LOG" >&2; exit 1; }
  echo "[ok] encoded $split"
done

echo
echo "=== 4. train, and score on the frozen-transfer evaluation pool ==="
for cell in $CELLS; do
  rep="${cell%%:*}"; learner="${cell#*:}"
  for seed in $SEEDS; do
    tag="${rep}__$(echo "$learner" | tr ':,' '__')__seed${seed}"
    out="$RUN_ROOT/cells/$tag"
    if [[ -f "$out/results.json" ]]; then echo "[skip] $tag done"; continue; fi
    hp=""
    first="$RUN_ROOT/cells/${rep}__$(echo "$learner" | tr ':,' '__')__seed$(echo $SEEDS | cut -d" " -f1)/results.json"
    # One hyperparameter search per cell, reused across its seeds: re-searching
    # per seed lets each pick the configuration suiting its own initialisation
    # and shrinks the very spread the seeds are there to measure.
    [[ "$seed" != "$(echo $SEEDS | cut -d' ' -f1)" && -f "$first" ]] && hp="--hp_from $first"
    python scripts/train_rep_learner_cell.py \
      --rep "$rep" --learner "$learner" \
      --prm_store "$REP_ROOT" --pb_store "$EVAL_REP_ROOT" \
      --train_stem train --val_stem val --test_stem val \
      --pb_subsets verifier generation \
      --out_dir "$out" --seed "$seed" --rescale zscore \
      $hp 2>&1 | tee -a "$LOG"
  done
done

echo
du -sh "$REP_ROOT"/* 2>/dev/null | tee -a "$LOG"
echo "[$(date)] reprobe_onpolicy done"
