#!/bin/bash
#SBATCH --job-name=s1ms_C_evalpb
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#
# Stage C: evaluate ProcessBench in parallel, one subset per GPU on a whole
# 4-GPU node (TamIA allocates h100 by node):
#   GPU0 -> gsm8k, GPU1 -> math, GPU2 -> olympiadbench, GPU3 -> omnimath.
# Each worker encodes its subset's dense hidden states (no truncation; OOM
# auto-retry) then scores them with the Stage-B probe at the PRM800K-val
# threshold and the per-subset 0.005 oracle grid (Sprint 1 convention).
#
# Required env: TAG
set -euo pipefail

# sbatch spools this script; use the launcher-exported S1MS_DIR (real repo path).
HERE="${S1MS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "$HERE/models.env"
# shellcheck disable=SC1091
source "$HERE/_common.sh"

: "${TAG:?TAG must be exported by the launcher}"
MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
BS="${S1MS_BATCH[$TAG]}"
MODEL_DIR="$RUNS_ROOT/$TAG"
export S1MS_MODEL_ID_CUR="$MODEL_ID" S1MS_STAGE="C_eval_pb"

LOG_DIR="$MODEL_DIR/logs"
mkdir -p "$LOG_DIR" "$MODEL_DIR/processbench_eval_shards"
LOG="$LOG_DIR/stageC_eval_pb.log"

s1ms_env_setup
echo "[C] TAG=$TAG model=$MODEL_ID 4-GPU fan-out (one subset/GPU) start_bs=$BS" | tee "$LOG"
s1ms_ensure_model_cached "$MODEL_ID" | tee -a "$LOG"
s1ms_venv

if [[ ! -f "$MODEL_DIR/linear_probe.pt" || ! -f "$MODEL_DIR/threshold.json" ]]; then
  echo "[C] FATAL: missing probe/threshold from Stage B in $MODEL_DIR" | tee -a "$LOG" >&2
  exit 1
fi

pids=()
for g in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="$g"
    SUBSET="${S1MS_SUBSETS[$g]}"
    export S1MS_SHARD="$SUBSET"
    RAW="$(s1ms_resolve_pb_raw "$SUBSET")"
    OUT="$MODEL_DIR/processbench_eval_shards/$SUBSET"
    WLOG="$LOG_DIR/stageC_eval_${SUBSET}.log"
    mkdir -p "$OUT"
    echo "[C] worker g=$g subset=$SUBSET raw=$RAW -> $OUT" | tee "$WLOG"

    # Encode this subset (no truncation; OOM auto-retry).
    run_with_oom_retry "$BS" "$WLOG" \
      python scripts/encode_processbench_hidden_states.py \
        --raw_file "$RAW" \
        --out_dir "$OUT" \
        --model_name_or_path "$MODEL_ID" --local_files_only \
        --run_name "s1ms_${TAG}_pb_${SUBSET}" \
        --max_seq_len -1 \
        --fail_on_overlength \
        --batch_size __BS__ \
        --model_dtype float16 --save_dtype float16 \
        --subset_name "$SUBSET" \
        --output_layout generic \
        --shard_idx 0 --num_shards 1 \
        --force

    # Score: val-selected threshold + 0.005 oracle.
    python scripts/evaluate_saved_probe_on_processbench.py \
      --probe "$MODEL_DIR/linear_probe.pt" \
      --pb_latents "$OUT/pb_step_h.npy" \
      --pb_meta "$OUT/pb_step_meta.jsonl" \
      --threshold_json "$MODEL_DIR/threshold.json" \
      --also_oracle --oracle_threshold_step 0.005 \
      --method_name dense_linear --pb_subset "$SUBSET" \
      --out_json "$OUT/metrics.json" \
      --out_scores_jsonl "$OUT/step_scores.jsonl" \
      --out_predictions_jsonl "$OUT/predictions.jsonl" 2>&1 | tee -a "$WLOG"
    echo "[C] subset $SUBSET done -> $OUT/metrics.json" | tee -a "$WLOG"
  ) &
  pids+=("$!")
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[C] FATAL: PB eval worker g=$i (subset=${S1MS_SUBSETS[$i]}) failed" | tee -a "$LOG" >&2
    fail=1
  fi
done
[[ $fail -ne 0 ]] && exit 1

echo "[C] all 4 subsets done -> $MODEL_DIR/processbench_eval_shards" | tee -a "$LOG"
