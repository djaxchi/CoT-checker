#!/bin/bash
#SBATCH --job-name=pb_full_store_7b
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# GO/NO-GO for the future-aware (lookahead) representation. Encode each
# ProcessBench solution as ONE full causal pass (all steps present, so downstream
# states have attended over earlier steps), then run the within-PB cross-validated
# ceiling test: current vs past+current vs past+current+future(W). No PRM re-encode
# yet; this only decides whether future context carries first-error signal.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
PB_DIR="${PB_DIR:-/scratch/d/dchikhi/cot-checker/processbench}"
REP_ROOT="$RUN_ROOT/repstore/pb_full_solution"
LOG_DIR="$RUN_ROOT/logs"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
SUBSETS="${SUBSETS:-gsm8k math olympiadbench omnimath}"

mkdir -p "$REP_ROOT" "$LOG_DIR"
export HF_HOME="$HF_CACHE"; export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
LOG_FILE="$LOG_DIR/pb_full_store_7b-${SLURM_JOB_ID:-$$}.log"

SPECS=()
for s in $SUBSETS; do
  f="$PB_DIR/processbench_${s}.jsonl"
  [[ -f "$f" ]] && SPECS+=("${s}:${f}") || echo "[warn] missing $f" | tee -a "$LOG_FILE"
done
[[ ${#SPECS[@]} -gt 0 ]] || { echo "[FATAL] no PB subsets"; exit 2; }
echo "[plan] $(date -Iseconds) specs: ${SPECS[*]}" | tee -a "$LOG_FILE"

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

CUDA_VISIBLE_DEVICES=0 python scripts/encode_processbench_full_store.py \
  --raw_specs "${SPECS[@]}" \
  --rep_root "$REP_ROOT" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --layer -1 --max_seq_len "$MAX_SEQ_LEN" --batch_size "$BATCH_SIZE" 2>&1 | tee -a "$LOG_FILE"

echo "[verify]" | tee -a "$LOG_FILE"; du -sh "$REP_ROOT"/* 2>/dev/null | tee -a "$LOG_FILE"

echo "===== LOOKAHEAD CEILING (within-PB group CV) =====" | tee -a "$LOG_FILE"
python scripts/analysis/lookahead_ceiling.py \
  --store_root "$REP_ROOT" --subsets $SUBSETS --windows 1 2 -1 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] pb_full_store_7b done"
