#!/bin/bash
#SBATCH --job-name=derive_delta_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# Derive the delta (transition) representation from the PRM800K token store into
# the dense-cache contract the harness consumes. Pure numpy over the memmapped
# store, no GPU. Run afterok the token-store encode.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
STORE_ROOT="$RUN_ROOT/repstore/tokens_last_layer"
OUT_CACHE="$RUN_ROOT/cache/qwen2_5_7b_delta"
LOG_DIR="$RUN_ROOT/logs"
SPLITS="${SPLITS:-probe_train_full val_5k test_2k}"

mkdir -p "$OUT_CACHE" "$LOG_DIR"
cd "$PROJECT_ROOT"
LOG_FILE="$LOG_DIR/derive_delta_7b-${SLURM_JOB_ID:-$$}.log"

echo "[derive_delta] store=$STORE_ROOT out=$OUT_CACHE splits=$SPLITS" | tee "$LOG_FILE"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy

python scripts/derive_delta_from_token_store.py \
  --store_root "$STORE_ROOT" \
  --splits $SPLITS \
  --out_dir "$OUT_CACHE" \
  --mode prm 2>&1 | tee -a "$LOG_FILE"

echo "[verify]" | tee -a "$LOG_FILE"
ls -la "$OUT_CACHE"/*.npy | tee -a "$LOG_FILE"
echo "[$(date)] derive_delta_7b done"
