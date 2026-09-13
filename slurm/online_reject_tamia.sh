#!/bin/bash
#SBATCH --job-name=online_reject
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

# Step-level rejection sampling: write one step, and rewrite it only if the
# checker condemns it.
#
# This exists because guided decoding (REPORT.md §20.9) lost to plain sampling by
# 0.091 at 10.7x the tokens, and the post-mortem blamed the procedure, not the
# head. The head ranked candidate steps well, +0.213 over choosing among the same
# five at random. What sank it was that branching needs diversity, so it ran at
# temperature 1.5, and that alone cost 0.304 before the checker acted; and that
# "lowest uncertainty of five" rewards steps that never commit, so it wrote 19.5
# steps against plain's 9.2 and often reached no answer at all.
#
# Rejection removes both. One step, at the policy's own temperature 1.0, so the
# sampler is untouched. The checker is only ever asked "is this step condemned",
# never "which of five is safest", so an ordinary committing step passes on the
# first draw. And a second draw is paid for only where the checker objects, so
# the cost is 1 + (rejection rate x retries) instead of a flat N.
#
# WHY THERE IS NO RANDOM ARM HERE. The guided run needed one because branching at
# raised temperature is a different sampler. This is not: accepting a uniformly
# chosen one of k i.i.d. draws from the policy IS one draw from the policy, so
# `plain` already is the exact checker-blind control. `reject_blind` runs the same
# retry machinery with the accept decision made by a coin at the measured
# rejection rate, which prices the loop and confirms it moves nothing by itself.
#
# TWO THRESHOLDS, NOT ONE. A single threshold that happened to work would not say
# whether the rule works, so the run sweeps two and reports both. They are set as
# quantiles of this cell's own offline step scores, since a raw number means
# something different for every head.
#
# STAGE 1 IS A GATE, for the same reason it is in online_bon_tamia.sh: if the
# online scoring path reconstructs spans even slightly differently from the
# encoder, every generated number is wrong without looking wrong.
#
# COST. Roughly 1,255 generation tokens per problem for plain and an estimated
# 1,800 for each rejection arm, against guided's 13,401, so the whole sweep is
# about a third of the guided job's load. The 5h wall clock is that ratio applied
# to the 8h guided allocation; shorten it on submit once the first shard reports.
#
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/reprobe_v1}"
ONPOLICY_ROOT="${ONPOLICY_ROOT:-$SCRATCH/cot_mech/onpolicy_v1}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
REP_ROOT="${REP_ROOT:-$RUN_ROOT/repstore/onpolicy_train_spans}"
CELL="${CELL:-$RUN_ROOT/cells/step_tokens__transformer_d256_l2_f1024_h4__seed42}"
# Held-out evaluation problems, disjoint from the ones the head trained on.
TRACES="${TRACES:-$ONPOLICY_ROOT/onpolicy_stage1_judge_traces.jsonl}"
# This cell's own offline step scores, which set the rejection threshold. It must
# be the same cell that does the scoring or the quantile means nothing.
CALIB="${CALIB:-$CELL/pb_step_scores_verifier.jsonl}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/online_reject_v1}"
LAYER="${LAYER:-35}"
MAX_PROBLEMS="${MAX_PROBLEMS:-300}"
# 16 truncated 42-61% of solutions in the guided run; 28 is what that settled on.
MAX_STEPS="${MAX_STEPS:-28}"
PLAIN_TEMPERATURE="${PLAIN_TEMPERATURE:-1.0}"
REJECT_TEMPERATURE="${REJECT_TEMPERATURE:-1.0}"
MAX_RETRIES="${MAX_RETRIES:-2}"
REJECT_QUANTILES="${REJECT_QUANTILES:-0.70 0.85}"
NUM_SHARDS="${NUM_SHARDS:-4}"

cd "$PROJECT_ROOT"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$HOME/venvs/cot/bin/activate"

mkdir -p "$OUT_DIR" "$RUN_ROOT/logs"
LOG="$RUN_ROOT/logs/online_reject-${SLURM_JOB_ID:-local}.log"
echo "cell=$CELL layer=$LAYER quantiles='$REJECT_QUANTILES' retries=$MAX_RETRIES" \
  | tee -a "$LOG"
git -C "$PROJECT_ROOT" rev-parse --short HEAD | tee -a "$LOG"

common=(--cell_dir "$CELL" --traces "$TRACES"
        --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only
        --layer "$LAYER" --prm_store "$REP_ROOT"
        --stats_cache "$RUN_ROOT/stats_cache")

# Run one arm across the four GPUs and wait for all of them.
run_sharded () {
  local stem="$1"; shift
  local pids=()
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/online_bon.py \
      "${common[@]}" "$@" \
      --max_steps "$MAX_STEPS" --max_problems "$MAX_PROBLEMS" \
      --shard_idx "$i" --num_shards "$NUM_SHARDS" \
      --out "$OUT_DIR/${stem}.shard${i}.jsonl" >>"$LOG" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || {
    echo "[FATAL] $stem failed" >&2; tail -40 "$LOG" >&2; exit 1; }
  echo "[ok] $stem"
}

echo
echo "=== 1. GATE: does online scoring reproduce this cell's offline scores? ==="
python scripts/onpolicy/online_bon.py "${common[@]}" \
  --verify_against "$CALIB" --verify_traces 20 2>&1 | tee -a "$LOG"
echo "[ok] scoring path verified"

echo
echo "=== 2. plain: the base policy, and the exact checker-blind control ==="
run_sharded plain --arms plain --plain_temperature "$PLAIN_TEMPERATURE"

for Q in $REJECT_QUANTILES; do
  tag="q${Q/./}"
  echo
  echo "=== 3.$tag reject: rewrite a step only when the checker condemns it ==="
  run_sharded "reject_${tag}" --arms reject \
    --reject_temperature "$REJECT_TEMPERATURE" --max_retries "$MAX_RETRIES" \
    --calibrate_from "$CALIB" --reject_quantile "$Q"

  # The blind arm has to spend what the real one spent, so its retry rate is the
  # rate the real one measured rather than the nominal quantile: a condemned step
  # is often replaced by another condemned step, and the nominal rate would
  # under-price the loop.
  RATE=$(python - "$OUT_DIR"/reject_${tag}.shard*.jsonl <<'PY'
import json, sys
n = d = 0
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        for a in r.get("attempts_per_step", []):
            n += 1
            d += a - 1
print(f"{(d / n) if n else 0.0:.4f}")
PY
)
  echo "[$tag] measured resample rate $RATE extra draws per step" | tee -a "$LOG"

  echo
  echo "=== 4.$tag reject_blind: the same loop, the coin flipped at random ==="
  run_sharded "blind_${tag}" --arms reject_blind \
    --reject_temperature "$REJECT_TEMPERATURE" --max_retries "$MAX_RETRIES" \
    --reject_tau 0 --blind_retry_rate "$RATE"

  echo
  echo "=== 5.$tag results ==="
  python scripts/analysis/online_bon_report.py \
    --rollouts "$OUT_DIR"/plain.shard*.jsonl \
               "$OUT_DIR"/reject_${tag}.shard*.jsonl \
               "$OUT_DIR"/blind_${tag}.shard*.jsonl \
    --outcomes "$ONPOLICY_ROOT/onpolicy_stage1_outcomes.jsonl" \
    --regrade_with "$TRACES" --max_steps "$MAX_STEPS" \
    --out "$RUN_ROOT/online_reject_report_${tag}.json" 2>&1 | tee -a "$LOG"
done

echo "[done] $OUT_DIR"
