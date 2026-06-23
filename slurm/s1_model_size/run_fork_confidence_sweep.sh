#!/bin/bash
#SBATCH --job-name=s1ms_forkconf
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# Stage 0 confidence sidecar: for the SAME sampled fork items already encoded by
# run_fork_encode_sweep.sh, record per-step model confidence (teacher-forced NLL /
# entropy / logit-gap over the candidate-step tokens) with every backbone. Output
# lands next to the hidden encode: runs/s1_model_size_dense/<tag>/forks/
#   forks_val_items_confidence.jsonl   (row-for-row aligned to *_meta.jsonl)
#
# This does NOT re-encode hidden states; it is the input to the question "is the
# correctness probe just model surprise?" (analyze via run_confidence_battery_tamia.sh).
#
# Uses the identical sampled set ($RUNS_ROOT/_forks_sample/forks_val_items.jsonl) so
# rows align with the existing meta. Knobs mirror run_fork_encode_sweep.sh:
#   S1MS_ONLY="qwen2_5_7b ..."   restrict / resume a subset
#   FORCE=1                      re-run completed sizes
set -uo pipefail

HERE="${S1MS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
if [[ ! -f "$HERE/models.env" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  HERE="$SLURM_SUBMIT_DIR/slurm/s1_model_size"
fi
# shellcheck disable=SC1091
source "$HERE/models.env"
# shellcheck disable=SC1091
source "$HERE/_common.sh"
set +e

STEM="forks_val_items"
SAMPLE="$RUNS_ROOT/_forks_sample/${STEM}.jsonl"

s1ms_env_setup
s1ms_venv

if [[ ! -f "$SAMPLE" ]]; then
  echo "[forkconf] FATAL: sampled fork set not found: $SAMPLE" >&2
  echo "[forkconf] Run run_fork_encode_sweep.sh first (it samples + encodes the forks)." >&2
  exit 1
fi

# shellcheck disable=SC2206
MODELS=( ${S1MS_ONLY:-${S1MS_TAGS[@]}} )
echo "[forkconf] models: ${MODELS[*]}  items=$SAMPLE"

fail=0
for TAG in "${MODELS[@]}"; do
  MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
  # confidence keeps full-vocab logits in memory; halve the encode batch to be safe.
  BS=$(( ${S1MS_BATCH[$TAG]} / 2 )); (( BS < 1 )) && BS=1
  OUT="$RUNS_ROOT/$TAG/forks"
  LOG_DIR="$RUNS_ROOT/$TAG/logs"
  mkdir -p "$OUT" "$LOG_DIR"
  LOG="$LOG_DIR/fork_confidence.log"
  export S1MS_MODEL_ID_CUR="$MODEL_ID" S1MS_STAGE="fork_confidence" S1MS_SHARD="0"

  if [[ ! -f "$OUT/${STEM}_h.npy" ]]; then
    echo "[forkconf] $TAG has no hidden encode yet; skipping (run fork encode first)" | tee -a "$LOG"
    continue
  fi
  if [[ -f "$OUT/${STEM}_confidence.jsonl" && "${FORCE:-0}" != "1" ]]; then
    echo "[forkconf] $TAG already has confidence; skipping (FORCE=1 to redo)" | tee -a "$LOG"
    continue
  fi
  echo "[forkconf] ===== $TAG ($MODEL_ID) start_bs=$BS =====" | tee "$LOG"
  s1ms_ensure_model_cached "$MODEL_ID" 2>&1 | tee -a "$LOG" || { fail=1; break; }

  CUDA_VISIBLE_DEVICES=0 run_with_oom_retry "$BS" "$LOG" \
    python scripts/encode_fork_confidence.py \
      --items "$SAMPLE" --out_dir "$OUT" --stem "$STEM" \
      --model_name_or_path "$MODEL_ID" --local_files_only \
      --run_name "s1ms_${TAG}_forkconf" \
      --max_seq_len -1 \
      --batch_size __BS__ --model_dtype float16 \
      --force
  if [[ $? -ne 0 ]]; then
    echo "[forkconf] $TAG FAILED; stopping (completed sizes preserved)." | tee -a "$LOG" >&2
    fail=1; break
  fi
  echo "[forkconf] $TAG done -> $OUT/${STEM}_confidence.jsonl" | tee -a "$LOG"
done

[[ $fail -ne 0 ]] && exit 1
echo "[forkconf] ALL DONE. Analyze: bash scripts/run_confidence_battery_tamia.sh"
