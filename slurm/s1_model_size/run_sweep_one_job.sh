#!/bin/bash
#SBATCH --job-name=s1ms_sweep
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#
# Whole-sweep single job: encode -> train -> eval -> aggregate for every model
# size, in order, on ONE 4-GPU allocation. Within a model the 4 GPUs run in
# parallel (PRM shards / PB subsets); models run strictly one at a time.
#
# Safeguards (this is the monolithic form, made as robust as a single job can be):
#   * 1.5B is the smoke gate: if it does not reproduce Sprint 1 within tolerance,
#     the loop STOPS before touching larger models.
#   * Each model's outputs persist to disk and refresh the leaderboard as soon as
#     it finishes, so progress is visible mid-run.
#   * Completed models are SKIPPED on re-run (resume after walltime/node death
#     without redoing finished sizes). Set FORCE=1 to recompute.
#   * A model failure stops the loop with earlier results intact (it does not
#     silently continue to the next size).
#
# Knobs:
#   sbatch --time=48:00:00 run_sweep_one_job.sh     # override walltime
#   S1MS_ONLY="qwen2_5_7b qwen2_5_14b" sbatch ...    # run/resume a subset
#   FORCE=1 sbatch ...                               # recompute completed models
set -uo pipefail   # deliberately NOT -e: per-model failures are handled explicitly

HERE="${S1MS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "$HERE/models.env"
# shellcheck disable=SC1091
source "$HERE/_common.sh"
set +e   # _common.sh enables `set -e`; turn it back off for the sweep loop

s1ms_env_setup
s1ms_venv

run_one_model() {
  local TAG="$1"
  local MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
  local PARAMS_LABEL="${S1MS_PARAMS_LABEL[$TAG]}"
  local BS="${S1MS_BATCH[$TAG]}"
  local MODEL_DIR="$RUNS_ROOT/$TAG"
  local MERGED="$MODEL_DIR/merged"
  local LOG_DIR="$MODEL_DIR/logs"
  export S1MS_MODEL_ID_CUR="$MODEL_ID" S1MS_STAGE="sweep"
  mkdir -p "$LOG_DIR" "$MERGED" \
           "$MODEL_DIR/prm800k_encode_shards" "$MODEL_DIR/processbench_eval_shards"
  local LOG="$LOG_DIR/sweep_${TAG}.log"

  if [[ -f "$MODEL_DIR/per_subset_metrics.json" && "${FORCE:-0}" != "1" ]]; then
    echo "[sweep] $TAG already complete; skipping (FORCE=1 to recompute)" | tee -a "$LOG"
    return 0
  fi

  echo "[sweep] ===== $TAG ($MODEL_ID) start_bs=$BS =====" | tee "$LOG"
  s1ms_ensure_model_cached "$MODEL_ID" 2>&1 | tee -a "$LOG" || return 1

  local f
  for f in "$PRM_TRAIN_JSONL" "$PRM_VAL_JSONL"; do
    [[ -f "$PRM_SPLIT_DIR/$f" ]] || {
      echo "[sweep] FATAL: missing frozen split $PRM_SPLIT_DIR/$f" | tee -a "$LOG" >&2; return 1; }
  done

  # ---- Pre-encode no-truncation audit (3B@32768 is the strict case).
  local PB_SPECS=() s raw
  for s in "${S1MS_SUBSETS[@]}"; do
    raw="$(s1ms_resolve_pb_raw "$s")" || return 1
    PB_SPECS+=("$s:$raw")
  done
  python scripts/s1ms_audit_token_lengths.py \
    --model_name_or_path "$MODEL_ID" --local_files_only \
    --prm_split_dir "$PRM_SPLIT_DIR" --prm_splits "$PRM_TRAIN_JSONL" "$PRM_VAL_JSONL" \
    --pb_subset "${PB_SPECS[@]}" --out_json "$MODEL_DIR/length_audit.json" \
    2>&1 | tee -a "$LOG" || return 1

  # ---- A: encode PRM800K (4-GPU fan-out; helper reads our locals via dynamic scope).
  s1ms_encode_prm_fanout || return 1

  # ---- B: merge shards + model_config + train probe + select threshold.
  for s in probe_train_40k val_1k; do
    python scripts/merge_prm800k_encoded_shards.py \
      --shard_root "$MODEL_DIR/prm800k_encode_shards" --stem "$s" \
      --out_dir "$MERGED" --force 2>&1 | tee -a "$LOG" || return 1
  done
  python scripts/s1ms_dump_model_config.py \
    --model_name_or_path "$MODEL_ID" --local_files_only \
    --params_label "$PARAMS_LABEL" --out_json "$MODEL_DIR/model_config.json" \
    2>&1 | tee -a "$LOG" || return 1
  python scripts/s1ms_train_dense_probe.py \
    --cache_dir "$MERGED" --out_dir "$MODEL_DIR" \
    --probe_train_stem probe_train_40k --val_stem val_1k \
    --model_name "$MODEL_ID" --seed 42 2>&1 | tee -a "$LOG" || return 1

  # ---- C: evaluate ProcessBench (4-GPU fan-out, one subset per GPU).
  s1ms_eval_pb_fanout || return 1

  # ---- D: aggregate (+ 1.5B gate) + refresh leaderboard.
  local GATE_ARGS=()
  if [[ "$TAG" == "qwen2_5_1_5b" ]]; then
    GATE_ARGS=( --expect_val_macro "$S1MS_EXPECT_VAL_MACRO"
                --expect_oracle_macro "$S1MS_EXPECT_ORACLE_MACRO"
                --tol "$S1MS_GATE_TOL" )
  fi
  python scripts/s1ms_aggregate_model.py \
    --eval_dir "$MODEL_DIR/processbench_eval_shards" --out_dir "$MODEL_DIR" \
    --subsets "${S1MS_SUBSETS[@]}" "${GATE_ARGS[@]}" 2>&1 | tee -a "$LOG" || return 1
  python scripts/s1ms_merge_leaderboard.py --runs_root "$RUNS_ROOT" 2>&1 | tee -a "$LOG" || true

  echo "[sweep] ===== $TAG done =====" | tee -a "$LOG"
  return 0
}

# Models to run (default: full sweep order; override/resume with S1MS_ONLY).
# shellcheck disable=SC2206
MODELS=( ${S1MS_ONLY:-${S1MS_TAGS[@]}} )
echo "[sweep] models: ${MODELS[*]}  (FORCE=${FORCE:-0})"

for TAG in "${MODELS[@]}"; do
  if ! run_one_model "$TAG"; then
    if [[ "$TAG" == "qwen2_5_1_5b" ]]; then
      echo "[sweep] FATAL: 1.5B smoke gate failed. Stopping; debug before scaling up." >&2
    else
      echo "[sweep] FATAL: $TAG failed. Stopping. Completed models are preserved;" >&2
      echo "[sweep] re-submit to resume (finished sizes are skipped)." >&2
    fi
    exit 1
  fi
done

echo "[sweep] ALL DONE -> $RUNS_ROOT/leaderboard_model_size.csv"
