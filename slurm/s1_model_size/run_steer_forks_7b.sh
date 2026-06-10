#!/bin/bash
#SBATCH --job-name=s1ms_steer7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# Step-1 causal steering test on the reserved 7B forks, at the strongest-signal
# layers from the depth sweep. For each layer: direction = error-mean minus
# correct-mean at that layer (from the multi-layer cache); steer toward correct
# with a forward hook; measure logp(correct)-logp(incorrect) on the held-out
# forks vs a random-direction control.
#
# Needs the layer-sweep job to have produced the multi-layer cache and the
# reserved steering_forks.jsonl. Knob: LAYERS (hidden_states indices).
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

TAG=qwen2_5_7b
MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
MDIR="$RUNS_ROOT/$TAG"
MLDIR="$MDIR/multilayer"
FORKS="$MDIR/steering/steering_forks.jsonl"
OUT="$MDIR/steering"
LOG_DIR="$MDIR/logs"
LAYERS="${LAYERS:-20 22}"
mkdir -p "$OUT" "$LOG_DIR"
LOG="$LOG_DIR/steer_forks.log"
export S1MS_MODEL_ID_CUR="$MODEL_ID" S1MS_STAGE="steer_forks" S1MS_SHARD="0"

s1ms_env_setup
echo "[steer] TAG=$TAG layers=$LAYERS forks=$FORKS" | tee "$LOG"
for need in "$MLDIR/multilayer_manifest.json" "$FORKS"; do
  [[ -e "$need" ]] || { echo "[steer] FATAL missing $need (run the layer sweep job first)" | tee -a "$LOG" >&2; exit 1; }
done
s1ms_ensure_model_cached "$MODEL_ID" 2>&1 | tee -a "$LOG" || exit 1
s1ms_venv
pip install --no-index matplotlib 2>&1 | tail -1 | tee -a "$LOG"

for li in $LAYERS; do
  echo "[steer] ===== layer hidden_states index $li =====" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0 python scripts/s1ms_steer_forks.py \
    --model_name_or_path "$MODEL_ID" --local_files_only \
    --ml_cache_dir "$MLDIR" --layer_index "$li" \
    --steering_forks "$FORKS" --out_dir "$OUT" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} -ne 0 ]] && { echo "[steer] FATAL: steering failed at L$li" | tee -a "$LOG" >&2; exit 1; }
done

mkdir -p "$RUNS_ROOT/figures"
cp "$OUT"/steer_forks_L*.png "$RUNS_ROOT/figures/" 2>/dev/null
echo "[steer] done -> $OUT (figures mirrored to $RUNS_ROOT/figures)" | tee -a "$LOG"
