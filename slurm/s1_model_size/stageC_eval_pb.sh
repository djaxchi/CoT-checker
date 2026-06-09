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

# ---- Encode + score the 4 subsets, one GPU each (shared helper in _common.sh).
s1ms_eval_pb_fanout || { echo "[C] FATAL: a PB subset worker failed" | tee -a "$LOG" >&2; exit 1; }
echo "[C] all 4 subsets done -> $MODEL_DIR/processbench_eval_shards" | tee -a "$LOG"
