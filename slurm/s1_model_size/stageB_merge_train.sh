#!/bin/bash
#SBATCH --job-name=s1ms_B_train
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Stage B: merge the 4 PRM800K encode shards, dump model_config.json, and train
# the DenseLinear probe + select the PRM800K-val threshold (Sprint 1 math,
# imported verbatim). CPU-only: a linear probe over the cached 40K hidden states
# does not need a GPU, so this stage does not hold a whole h100 node.
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
PARAMS_LABEL="${S1MS_PARAMS_LABEL[$TAG]}"
MODEL_DIR="$RUNS_ROOT/$TAG"
MERGED="$MODEL_DIR/merged"
SHARD_ROOT="$MODEL_DIR/prm800k_encode_shards"
LOG_DIR="$MODEL_DIR/logs"
mkdir -p "$MERGED" "$LOG_DIR"
LOG="$LOG_DIR/stageB_merge_train.log"
export S1MS_STAGE="B_merge_train"

s1ms_env_setup
echo "[B] TAG=$TAG model=$MODEL_ID merge+train" | tee "$LOG"
s1ms_venv

# ---- Merge PRM800K shards back into single deterministic splits.
for stem in probe_train_40k val_1k; do
  python scripts/merge_prm800k_encoded_shards.py \
    --shard_root "$SHARD_ROOT" --stem "$stem" --out_dir "$MERGED" --force \
    2>&1 | tee -a "$LOG"
done

# ---- Model architecture metadata (no weights loaded).
python scripts/s1ms_dump_model_config.py \
  --model_name_or_path "$MODEL_ID" --local_files_only \
  --params_label "$PARAMS_LABEL" \
  --out_json "$MODEL_DIR/model_config.json" 2>&1 | tee -a "$LOG"

# ---- Train DenseLinear probe + select PRM800K-val threshold (Sprint 1).
python scripts/s1ms_train_dense_probe.py \
  --cache_dir "$MERGED" --out_dir "$MODEL_DIR" \
  --probe_train_stem probe_train_40k --val_stem val_1k \
  --model_name "$MODEL_ID" --seed 42 2>&1 | tee -a "$LOG"

echo "[B] done -> $MODEL_DIR/linear_probe.pt + threshold.json" | tee -a "$LOG"
