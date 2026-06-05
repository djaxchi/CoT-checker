#!/bin/bash
# ---------------------------------------------------------------------------
# Sprint 2: build the full-scale PRM800K fork dataset.
#
# CPU/IO only (tokenizer needed for the length filter; NO GPU). Run directly on
# the login node, or inside an interactive CPU allocation if cluster policy
# requires it (e.g. `salloc --cpus-per-task=16 --mem=64G --time=3:00:00`).
# Do NOT submit this to the GPU queue.
#
# Produces, under $FORKS_DIR: forks_full.jsonl, forks_{train,val}_items.jsonl,
# forks_{train,val}_pairs.jsonl, forks_manifest.json.
#
# Run:  bash scripts/tamia/jobs/s2/build_forks.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_BASE="$SCRATCH/cot_mech"
RAW_FILE="$SCRATCH_BASE/raw/prm800k/train.jsonl"  # train split only (matches prestudy source; no test leakage)
FORKS_DIR="$SCRATCH_BASE/s2_forks/data"
MODEL="Qwen/Qwen2.5-1.5B"

N_TRAIN_FORKS="${N_TRAIN_FORKS:-40000}"
N_VAL_FORKS="${N_VAL_FORKS:-5000}"
PAIR_MODE="${PAIR_MODE:-one}"

cd "$PROJECT_DIR"
mkdir -p "$FORKS_DIR"

# CPU-only environment (no cuda module needed).
module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11
source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "======================================================="
echo "  S2: build forks  (mode=$PAIR_MODE  train=$N_TRAIN_FORKS  val=$N_VAL_FORKS)"
echo "  raw : $RAW_FILE"
echo "  out : $FORKS_DIR"
echo "======================================================="

python scripts/build_prm800k_forks.py \
    --raw_file "$RAW_FILE" \
    --out_dir "$FORKS_DIR" \
    --tokenizer_name_or_path "$MODEL" \
    --local_files_only \
    --run_name s2_forks \
    --n_train_forks "$N_TRAIN_FORKS" \
    --n_val_forks "$N_VAL_FORKS" \
    --pair_mode "$PAIR_MODE"

echo ""
echo "  Done. Inspect stats with: bash scripts/tamia/jobs/s2/stats_forks.sh"
