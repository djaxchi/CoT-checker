#!/bin/bash
#SBATCH --job-name=s1ms_lsweep7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#
# Phase 1 layer sweep on Qwen2.5-7B:
#   1. reserve a small held-out fork set for the later steering experiment
#   2. multi-layer encode PRM800K 40k/1k (last-token state at each decile layer)
#   3. per-layer correctness probe + minimal-subset sweep -> F1-vs-depth
#
# Knobs: FORK_ITEMS (default S2 forks_val_items.jsonl), N_FORKS (default 15),
#        LAYER_FRACS (default deciles).
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
BS="${S1MS_BATCH[$TAG]}"
MDIR="$RUNS_ROOT/$TAG"
MLDIR="$MDIR/multilayer"
LOG_DIR="$MDIR/logs"
FORK_ITEMS="${FORK_ITEMS:-/scratch/d/dchikhi/cot_mech/s2_forks/data/forks_val_items.jsonl}"
N_FORKS="${N_FORKS:-15}"
LAYER_FRACS="${LAYER_FRACS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
mkdir -p "$MLDIR" "$LOG_DIR" "$MDIR/steering"
LOG="$LOG_DIR/layer_sweep.log"
export S1MS_MODEL_ID_CUR="$MODEL_ID" S1MS_STAGE="layer_sweep" S1MS_SHARD="0"

s1ms_env_setup
echo "[lsweep] TAG=$TAG model=$MODEL_ID forks=$FORK_ITEMS" | tee "$LOG"
s1ms_ensure_model_cached "$MODEL_ID" 2>&1 | tee -a "$LOG" || exit 1
s1ms_venv
# layer sweep also needs sklearn
pip install --no-index scikit-learn 2>&1 | tail -1 | tee -a "$LOG"

# Frozen PRM split must be present.
for f in "$PRM_TRAIN_JSONL" "$PRM_VAL_JSONL"; do
  [[ -f "$PRM_SPLIT_DIR/$f" ]] || { echo "[lsweep] FATAL missing $PRM_SPLIT_DIR/$f" | tee -a "$LOG" >&2; exit 1; }
done

# 1. reserve steering forks (CPU; non-fatal if fork items absent, just warn)
if [[ -f "$FORK_ITEMS" ]]; then
  python scripts/s1ms_reserve_steering_forks.py --items "$FORK_ITEMS" \
    --n_forks "$N_FORKS" --seed 7 --out "$MDIR/steering/steering_forks.jsonl" 2>&1 | tee -a "$LOG"
else
  echo "[lsweep] WARNING: FORK_ITEMS not found ($FORK_ITEMS); skipping fork reservation." | tee -a "$LOG" >&2
fi

# 2. multi-layer encode PRM800K 40k/1k (single GPU; 7B fits comfortably)
CUDA_VISIBLE_DEVICES=0 run_with_oom_retry "$BS" "$LOG" \
  python scripts/encode_prm800k_multilayer.py \
    --data_dir "$PRM_SPLIT_DIR" --out_dir "$MLDIR" \
    --model_name_or_path "$MODEL_ID" --local_files_only \
    --run_name "s1ms_${TAG}_multilayer" \
    --splits "${PRM_TRAIN_JSONL}:probe_train_40k" "${PRM_VAL_JSONL}:val_1k" \
    --layer_fracs $LAYER_FRACS \
    --max_seq_len -1 --batch_size __BS__ --model_dtype float16 --save_dtype float16 --force
[[ $? -ne 0 ]] && { echo "[lsweep] FATAL: multi-layer encode failed" | tee -a "$LOG" >&2; exit 1; }

# 3. per-layer probe + subset sweep (CPU)
python scripts/s1ms_layer_sweep.py \
  --cache_dir "$MLDIR" --out_dir "$MDIR/layer_sweep" 2>&1 | tee -a "$LOG"
[[ $? -ne 0 ]] && { echo "[lsweep] FATAL: layer sweep failed" | tee -a "$LOG" >&2; exit 1; }

# Mirror figures into the shared figures dir for easy rsync.
mkdir -p "$RUNS_ROOT/figures"
cp "$MDIR/layer_sweep/"*.png "$RUNS_ROOT/figures/" 2>/dev/null
echo "[lsweep] done. figures in $MDIR/layer_sweep and $RUNS_ROOT/figures" | tee -a "$LOG"
