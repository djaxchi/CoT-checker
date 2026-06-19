#!/bin/bash
#SBATCH --job-name=prm800k_heldout_allsizes
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=%x-%A_%a.out

# Encode the PRM800K held-out TEST set with EACH model-size backbone (last layer /
# last token = the deployed readout), so every size's dense probe can be evaluated
# on never-seen data in ITS OWN hidden space. One array task per size (independent
# single-GPU jobs -> backfill quickly instead of one big reservation).
#
# Writes <RUNS_ROOT>/<tag>/merged/prm800k_heldout_test_{h,y,meta}.
# Then eval locally with scripts/eval_prm800k_heldout_probe.py.
#
# Usage:
#   sbatch slurm/encode_prm800k_heldout_allsizes_tamia.sh                 # all 5 sizes
#   sbatch --array=2 slurm/encode_prm800k_heldout_allsizes_tamia.sh       # just 7B
#   STEM=prm800k_heldout_test FORCE=1 sbatch ...

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
cd "$PROJECT_ROOT"
source slurm/s1_model_size/models.env   # S1MS_TAGS, S1MS_MODEL_ID, S1MS_BATCH, RUNS_ROOT, PRM_SPLIT_DIR

TAG="${S1MS_TAGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
BATCH="${S1MS_BATCH[$TAG]}"
STEM="${STEM:-prm800k_heldout_test}"
DATA_DIR="${DATA_DIR:-$PRM_SPLIT_DIR}"
OUT_DIR="$RUNS_ROOT/$TAG/merged"

export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT_DIR"

cat <<BANNER
================================================================
job      : ${SLURM_JOB_NAME} ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID:-N/A}
host     : $(hostname)   date $(date -Iseconds)
tag      : $TAG   model: $MODEL_ID   batch: $BATCH
data_dir : $DATA_DIR/${STEM}.jsonl
out_dir  : $OUT_DIR
================================================================
BANNER

if [[ ! -f "$DATA_DIR/${STEM}.jsonl" ]]; then
  echo "[FATAL] missing $DATA_DIR/${STEM}.jsonl (build it first)"; exit 1
fi

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy scikit-learn

EXTRA=()
if [[ "${FORCE:-0}" == "1" ]]; then EXTRA+=(--force); fi

python scripts/encode_prm800k_hidden_states.py \
  --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
  --model_name_or_path "$MODEL_ID" --local_files_only \
  --run_name "heldout_test_${TAG}" --max_seq_len -1 \
  --batch_size "$BATCH" --model_dtype float16 --save_dtype float16 \
  --splits "${STEM}.jsonl:${STEM}" "${EXTRA[@]}"

echo "[$(date -Iseconds)] $TAG encode done -> $OUT_DIR/${STEM}_{h,y,meta}"

# ---- eval this size's deployed probe on the just-encoded test (last layer) ----
# Per-tag JSON filename so the 5 parallel array tasks never race on one file.
RUN_DIR="$RUNS_ROOT/$TAG"
LABEL="${S1MS_PARAMS_LABEL[$TAG]:-$TAG}"
EVAL_OUT="${EVAL_OUT:-$PROJECT_ROOT/results/prm800k_test_full_eval}"
if [[ -f "$RUN_DIR/linear_probe.pt" ]]; then
  echo "[$(date -Iseconds)] $TAG eval -> $EVAL_OUT/${LABEL}.json"
  python scripts/eval_prm800k_heldout_probe.py \
    --run_dir "$RUN_DIR" --enc_dir "$OUT_DIR" --stem "$STEM" \
    --tag "$LABEL" --out_dir "$EVAL_OUT"
else
  echo "[WARN] $TAG: no $RUN_DIR/linear_probe.pt; skipping eval (encoding kept)"
fi

echo "[$(date -Iseconds)] $TAG done"
