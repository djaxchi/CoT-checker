#!/bin/bash
#SBATCH --job-name=s1ms_forkenc
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# Encode a sampled set of PRM800K fork siblings (anchor / positive / negative)
# with every backbone, so the matched-fork geometry can be compared across model
# size. The SAME sampled item set is encoded by all sizes (identical forks), then
# each size's hidden states land in runs/s1_model_size_dense/<tag>/forks/.
#
# Visualize with: python scripts/s1ms_viz_representations.py --source forks
#
# Knobs:
#   FORK_ITEMS=/path/forks_val_items.jsonl   # source fork items (default below)
#   N_FORKS=1000                             # how many forks to sample
#   S1MS_ONLY="qwen2_5_7b ..."               # restrict / resume a subset
#   FORCE=1                                  # re-encode completed sizes
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

FORK_ITEMS="${FORK_ITEMS:-/scratch/d/dchikhi/cot_mech/s2_forks/data/forks_val_items.jsonl}"
N_FORKS="${N_FORKS:-1000}"
STEM="forks_val_items"
SAMPLE="$RUNS_ROOT/_forks_sample/${STEM}.jsonl"

s1ms_env_setup
s1ms_venv

if [[ ! -f "$FORK_ITEMS" ]]; then
  echo "[forkenc] FATAL: fork items not found: $FORK_ITEMS" >&2
  echo "[forkenc] Set FORK_ITEMS to the S2 forks_val_items.jsonl (built by build_prm800k_forks.py)." >&2
  exit 1
fi

# Sample once (shared across all sizes) unless already present.
if [[ ! -f "$SAMPLE" || "${FORCE:-0}" == "1" ]]; then
  python scripts/s1ms_sample_forks.py --items "$FORK_ITEMS" \
    --n_forks "$N_FORKS" --seed 42 --out "$SAMPLE" || exit 1
fi

# shellcheck disable=SC2206
MODELS=( ${S1MS_ONLY:-${S1MS_TAGS[@]}} )
echo "[forkenc] models: ${MODELS[*]}  items=$SAMPLE"

fail=0
for TAG in "${MODELS[@]}"; do
  MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
  BS="${S1MS_BATCH[$TAG]}"
  OUT="$RUNS_ROOT/$TAG/forks"
  LOG_DIR="$RUNS_ROOT/$TAG/logs"
  mkdir -p "$OUT" "$LOG_DIR"
  LOG="$LOG_DIR/fork_encode.log"
  export S1MS_MODEL_ID_CUR="$MODEL_ID" S1MS_STAGE="fork_encode" S1MS_SHARD="0"

  if [[ -f "$OUT/${STEM}_h.npy" && "${FORCE:-0}" != "1" ]]; then
    echo "[forkenc] $TAG already encoded; skipping (FORCE=1 to redo)" | tee -a "$LOG"
    continue
  fi
  echo "[forkenc] ===== $TAG ($MODEL_ID) start_bs=$BS =====" | tee "$LOG"
  s1ms_ensure_model_cached "$MODEL_ID" 2>&1 | tee -a "$LOG" || { fail=1; break; }

  CUDA_VISIBLE_DEVICES=0 run_with_oom_retry "$BS" "$LOG" \
    python scripts/encode_prm800k_forks.py \
      --items "$SAMPLE" --out_dir "$OUT" --stem "$STEM" \
      --model_name_or_path "$MODEL_ID" --local_files_only \
      --run_name "s1ms_${TAG}_forks" \
      --max_seq_len -1 \
      --batch_size __BS__ --model_dtype float16 --save_dtype float16 \
      --force
  if [[ $? -ne 0 ]]; then
    echo "[forkenc] $TAG FAILED; stopping (completed sizes preserved)." | tee -a "$LOG" >&2
    fail=1; break
  fi
  echo "[forkenc] $TAG done -> $OUT/${STEM}_h.npy" | tee -a "$LOG"
done

[[ $fail -ne 0 ]] && exit 1
echo "[forkenc] ALL DONE. Visualize: python scripts/s1ms_viz_representations.py --source forks"
