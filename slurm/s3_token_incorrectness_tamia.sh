#!/bin/bash
#SBATCH --job-name=s3_token_traj
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# Per-token re-encode of the PRM800K 6k held-out test: apply the L28 dense probe to
# EVERY step token (+ L20 diagnostic) with the model's per-token certainty, then plot
# the token heatmap / spike / certainty-coincidence figures. Same items + tokenization
# as the §15 encode, just keeping the full span instead of {first,last}.
#
# Usage:
#   sbatch slurm/s3_token_incorrectness_tamia.sh
#   FORCE=1 LIMIT=200 sbatch slurm/s3_token_incorrectness_tamia.sh   # quick smoke

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
DATA_DIR="${DATA_DIR:-$SCRATCH/cot_mech/prestudy_v1/data}"
ITEMS="${ITEMS:-$DATA_DIR/prm800k_heldout_test.jsonl}"
PROBE="${PROBE:-$PROJECT_ROOT/runs/s1_model_size_dense/qwen2_5_7b/linear_probe.pt}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/runs/s3_token_traj/qwen2_5_7b}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
STEM="${STEM:-prm800k_heldout_test}"
LAYERS="${LAYERS:-20 28}"
PROBE_LAYER="${PROBE_LAYER:-28}"
BATCH_SIZE="${BATCH_SIZE:-8}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"

mkdir -p "$OUT_DIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-s3_token_traj}  id ${SLURM_JOB_ID:-N/A}
host       : $(hostname)   date $(date -Iseconds)
git_commit : $GIT_COMMIT
model      : $MODEL_NAME_OR_PATH
items      : $ITEMS
probe      : $PROBE  (native L$PROBE_LAYER)
layers     : $LAYERS
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

python scripts/analysis/s3_token_incorrectness_extract.py \
  --items "$ITEMS" --probe "$PROBE" --out_dir "$OUT_DIR" --stem "$STEM" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --layers $LAYERS --probe_layer "$PROBE_LAYER" \
  --batch_size "$BATCH_SIZE" --max_seq_len -1 "${EXTRA[@]}"

# Plotting is cheap + non-essential; never let it fail the compute job.
python scripts/analysis/s3_token_incorrectness_plot.py \
  --tokens "$OUT_DIR/${STEM}_tokens.jsonl" \
  --out "$OUT_DIR/plots" --probe_layer "$PROBE_LAYER" --n_examples 9 \
  || echo "[warn] plotting failed; rerun locally on ${STEM}_tokens.jsonl"

echo "[$(date)] token-trajectory done -> $OUT_DIR"
