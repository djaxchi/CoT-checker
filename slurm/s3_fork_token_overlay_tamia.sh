#!/bin/bash
#SBATCH --job-name=s3_fork_traj
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Per-token re-encode of the PRM800K FORK items (matched correct-vs-incorrect next steps
# sharing one problem+prefix), applying the L28 dense probe to every step token with the
# model's per-token confidence and dense representation stats, then overlaying each fork's
# correct vs incorrect step trajectories. Same extractor as the 6k heldout token-traj job;
# only the item file (forks_val_items.jsonl) and the plot script differ.
#
# TamIA allocates h100 by whole node -> take all 4 GPUs, shard items across them, cat the
# per-shard jsonl (fork items are independent rows; the overlay plot regroups by fork_id
# from the merged file), then plot. Cap to MAX_FORKS distinct forks to keep it fast.
#
# Usage:
#   sbatch slurm/s3_fork_token_overlay_tamia.sh
#   MAX_FORKS=300 sbatch slurm/s3_fork_token_overlay_tamia.sh          # smaller/faster
#   FORCE=1 LIMIT=200 sbatch slurm/s3_fork_token_overlay_tamia.sh      # quick smoke

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
FORKS_DIR="${FORKS_DIR:-$SCRATCH/cot_mech/s2_forks/data}"
ITEMS="${ITEMS:-$FORKS_DIR/forks_val_items.jsonl}"
PROBE="${PROBE:-$PROJECT_ROOT/runs/s1_model_size_dense/qwen2_5_7b/linear_probe.pt}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/runs/s3_fork_traj/qwen2_5_7b}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
STEM="${STEM:-forks_val_items}"
LAYERS="${LAYERS:-20 28}"
PROBE_LAYER="${PROBE_LAYER:-28}"
MAX_FORKS="${MAX_FORKS:-1000}"
ACTIVE_TAU="${ACTIVE_TAU:-6.0}"
N_FORKS="${N_FORKS:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# The 7B weights are cached under the S1 HF cache (models.env HF_CACHE_ROOT), NOT
# $SCRATCH/hf_cache; pointing HF_HOME elsewhere makes offline mode miss the model.
source "$PROJECT_ROOT/slurm/s1_model_size/models.env"
mkdir -p "$OUT_DIR"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-s3_fork_traj}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
git_commit : $GIT_COMMIT
model      : $MODEL_NAME_OR_PATH
items      : $ITEMS
probe      : $PROBE  (native L$PROBE_LAYER)
layers     : $LAYERS   max_forks $MAX_FORKS   active_tau $ACTIVE_TAU
out_dir    : $OUT_DIR
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy matplotlib

EXTRA=()
[[ "${FORCE:-0}" == "1" ]] && EXTRA+=(--force)
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")

NUM_SHARDS="${NUM_SHARDS:-4}"
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="$i" python scripts/analysis/s3_token_incorrectness_extract.py \
    --items "$ITEMS" --probe "$PROBE" --out_dir "$OUT_DIR" --stem "$STEM" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --layers $LAYERS --probe_layer "$PROBE_LAYER" \
    --max_forks "$MAX_FORKS" --active_tau "$ACTIVE_TAU" \
    --batch_size "$BATCH_SIZE" --max_seq_len -1 \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" "${EXTRA[@]}" \
    > "$OUT_DIR/extract_shard${i}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ $fail -ne 0 ]]; then
  echo "[FATAL] a shard failed; tails below:"; tail -n 40 "$OUT_DIR"/extract_shard*.log
  exit 1
fi

# Merge shards (fork items are independent rows; the plot regroups by fork_id).
cat "$OUT_DIR/${STEM}_shard"*"_tokens.jsonl" > "$OUT_DIR/${STEM}_tokens.jsonl"
cat "$OUT_DIR/${STEM}_shard"*"_steps.jsonl"  > "$OUT_DIR/${STEM}_steps.jsonl"
echo "[merge] $(wc -l < "$OUT_DIR/${STEM}_tokens.jsonl") token rows, "\
"$(wc -l < "$OUT_DIR/${STEM}_steps.jsonl") steps"

# Plotting is cheap + non-essential; never let it fail the compute job.
python scripts/analysis/s3_fork_token_overlay_plot.py \
  --tokens "$OUT_DIR/${STEM}_tokens.jsonl" \
  --out "$OUT_DIR/plots" --layer "$PROBE_LAYER" --n_forks "$N_FORKS" \
  || echo "[warn] plotting failed; rerun locally on ${STEM}_tokens.jsonl"

echo "[$(date)] fork token-trajectory done -> $OUT_DIR"
