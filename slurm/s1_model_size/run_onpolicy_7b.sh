#!/bin/bash
#SBATCH --job-name=s1ms_onpolicy
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# Stage 1 (decisive control): generate the model's OWN reasoning steps, grade them by
# final-answer match, encode them in the probe's readout, and test whether the probe
# separates correct vs incorrect ON-POLICY steps (uniformly low perplexity). If it
# does, the probe reads correctness, not surprise / distribution.
#
# Four GPU/CPU stages chained for one size (default 7B):
#   1. generate_onpolicy_steps.py   -> onpolicy_val_items.jsonl (+ trajectories)
#   2. encode_prm800k_forks.py      -> onpolicy_val_h.npy / _meta.jsonl
#   3. encode_fork_confidence.py    -> onpolicy_val_confidence.jsonl (NLL sanity)
#   4. analyze_onpolicy_probe.py    -> runs/fork_rep_audit/<tag>/onpolicy/
#
# SMOKE FIRST: bash with N_PROBLEMS=20 to confirm the base model boxes answers and
# produces both correct AND incorrect trajectories before the full run.
#
# Knobs: TAG, N_PROBLEMS, N_SAMPLES, TEMPERATURE, MAX_NEW_TOKENS
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

TAG="${TAG:-qwen2_5_7b}"
N_PROBLEMS="${N_PROBLEMS:-300}"
N_SAMPLES="${N_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
STEM="onpolicy_val"
SAMPLE="$RUNS_ROOT/_forks_sample/forks_val_items.jsonl"
MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
ENC_BS="${S1MS_BATCH[$TAG]}"
OUT="$RUNS_ROOT/$TAG/onpolicy"
PROBE="$RUNS_ROOT/$TAG/linear_probe.pt"
ANALYSIS_OUT="$RUNS_ROOT/../fork_rep_audit/$TAG/onpolicy"
FORK_CONF="$RUNS_ROOT/$TAG/forks/forks_val_items_confidence.jsonl"

s1ms_env_setup
s1ms_venv
mkdir -p "$OUT"

[[ -f "$SAMPLE" ]] || { echo "[onpolicy] FATAL: sampled forks not found: $SAMPLE"; exit 1; }
[[ -f "$PROBE"  ]] || { echo "[onpolicy] FATAL: probe not found: $PROBE"; exit 1; }
s1ms_ensure_model_cached "$MODEL_ID" || exit 1

echo "[onpolicy] === 1/4 generate ($TAG, $N_PROBLEMS x $N_SAMPLES, T=$TEMPERATURE) ==="
python scripts/generate_onpolicy_steps.py \
  --fork_items "$SAMPLE" --out_dir "$OUT" --stem "$STEM" \
  --model_name_or_path "$MODEL_ID" --local_files_only \
  --run_name "s1ms_${TAG}_onpolicy" \
  --n_samples "$N_SAMPLES" --temperature "$TEMPERATURE" \
  --max_new_tokens "$MAX_NEW_TOKENS" --max_problems "$N_PROBLEMS" --force \
  || { echo "[onpolicy] generation FAILED"; exit 1; }

ITEMS="$OUT/${STEM}_items.jsonl"
NITEMS=$(wc -l < "$ITEMS" 2>/dev/null || echo 0)
[[ "$NITEMS" -gt 0 ]] || { echo "[onpolicy] FATAL: no step items produced (base model not boxing answers?)"; exit 1; }

echo "[onpolicy] === 2/4 encode hidden states ($NITEMS items) ==="
CUDA_VISIBLE_DEVICES=0 run_with_oom_retry "$ENC_BS" "$OUT/encode.log" \
  python scripts/encode_prm800k_forks.py \
    --items "$ITEMS" --out_dir "$OUT" --stem "$STEM" \
    --model_name_or_path "$MODEL_ID" --local_files_only \
    --run_name "s1ms_${TAG}_onpolicy_enc" --max_seq_len -1 \
    --batch_size __BS__ --model_dtype float16 --save_dtype float16 --force \
  || { echo "[onpolicy] encode FAILED"; exit 1; }

echo "[onpolicy] === 3/4 confidence sidecar (NLL sanity) ==="
CONF_BS=$(( ENC_BS / 2 )); (( CONF_BS < 1 )) && CONF_BS=1
CUDA_VISIBLE_DEVICES=0 run_with_oom_retry "$CONF_BS" "$OUT/confidence.log" \
  python scripts/encode_fork_confidence.py \
    --items "$ITEMS" --out_dir "$OUT" --stem "$STEM" \
    --model_name_or_path "$MODEL_ID" --local_files_only \
    --run_name "s1ms_${TAG}_onpolicy_conf" --max_seq_len -1 \
    --batch_size __BS__ --model_dtype float16 --force \
  || echo "[onpolicy] WARN: confidence sidecar failed (NLL sanity will be skipped)"

echo "[onpolicy] === 4/4 analysis ==="
CONF_ARG=(); [[ -f "$OUT/${STEM}_confidence.jsonl" ]] && CONF_ARG=(--confidence "$OUT/${STEM}_confidence.jsonl")
FNEG_ARG=(); [[ -f "$FORK_CONF" ]] && FNEG_ARG=(--fork_neg_confidence "$FORK_CONF")
python scripts/analyze_onpolicy_probe.py \
  --h "$OUT/${STEM}_h.npy" --meta "$OUT/${STEM}_meta.jsonl" --items "$ITEMS" \
  --probe "$PROBE" "${CONF_ARG[@]}" "${FNEG_ARG[@]}" --out "$ANALYSIS_OUT" \
  || { echo "[onpolicy] analysis FAILED"; exit 1; }

echo "[onpolicy] DONE. Report: cat $ANALYSIS_OUT/onpolicy_report.md"
cat "$ANALYSIS_OUT/onpolicy_report.md" 2>/dev/null
