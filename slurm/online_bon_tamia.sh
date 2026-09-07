#!/bin/bash
#SBATCH --job-name=online_bon
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

# ReProbe's ONLINE mode: the checker chooses the next step while the solution is
# being written, rather than reranking finished solutions.
#
#     Q_online(r_t) = 1 - U(r_t | r_<t, x)
#
# Three arms on the same problems and seeds, so the comparison is paired:
#
#   plain    one step per position                    the base policy
#   random   N candidates, uniform choice             search without the checker
#   guided   N candidates, lowest uncertainty wins    search with the checker
#
# The random arm is the whole point. Branching N ways and keeping any one of them
# is already a different sampler, so guided-beats-plain would not show the head
# contributed anything. random -> guided is the only contrast that isolates it.
#
# STAGE 1 IS A GATE. The head was trained on step spans encoded under
# verifier_prefix at one layer; if the online scoring path reconstructs those
# spans even slightly differently, every generated number would be wrong without
# looking wrong. So the run first re-scores stored trajectories and requires
# agreement with the offline scores the cell already wrote, and stops if it
# disagrees. Generation is far too expensive to spend on an unverified scorer.
#
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/reprobe_v1}"
ONPOLICY_ROOT="${ONPOLICY_ROOT:-$SCRATCH/cot_mech/onpolicy_v1}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
REP_ROOT="${REP_ROOT:-$RUN_ROOT/repstore/onpolicy_train_spans}"
CELL="${CELL:-$RUN_ROOT/cells/step_tokens__transformer_d256_l2_f1024_h4__seed42}"
TRACES="${TRACES:-$ONPOLICY_ROOT/onpolicy_stage1_judge_traces.jsonl}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/online_bon}"
LAYER="${LAYER:-35}"
N_CANDIDATES="${N_CANDIDATES:-5}"
TEMPERATURE="${TEMPERATURE:-1.5}"
MAX_PROBLEMS="${MAX_PROBLEMS:-300}"
MAX_STEPS="${MAX_STEPS:-16}"
NUM_SHARDS="${NUM_SHARDS:-4}"
ARMS="${ARMS:-plain random guided}"

cd "$PROJECT_ROOT"
[[ -f models.env ]] && source models.env
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_CACHE_ROOT:-$SCRATCH/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$HOME/venvs/cot/bin/activate"

mkdir -p "$OUT_DIR" "$RUN_ROOT/logs"
LOG="$RUN_ROOT/logs/online_bon-${SLURM_JOB_ID:-local}.log"
echo "cell=$CELL layer=$LAYER N=$N_CANDIDATES T=$TEMPERATURE arms='$ARMS'" | tee -a "$LOG"
git -C "$PROJECT_ROOT" rev-parse --short HEAD | tee -a "$LOG"

echo
echo "=== 1. GATE: does online scoring reproduce this cell's offline scores? ==="
python scripts/onpolicy/online_bon.py \
  --cell_dir "$CELL" --traces "$TRACES" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --layer "$LAYER" --prm_store "$REP_ROOT" \
  --stats_cache "$RUN_ROOT/stats_cache" \
  --verify_against "$CELL/pb_step_scores_verifier.jsonl" \
  --verify_traces 20 2>&1 | tee -a "$LOG"
echo "[ok] scoring path verified"

echo
echo "=== 2. generate: $ARMS, sharded over $NUM_SHARDS GPUs ==="
pids=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/online_bon.py \
    --cell_dir "$CELL" --traces "$TRACES" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --layer "$LAYER" --prm_store "$REP_ROOT" \
    --stats_cache "$RUN_ROOT/stats_cache" \
    --arms $ARMS --n_candidates "$N_CANDIDATES" \
    --temperature "$TEMPERATURE" --max_steps "$MAX_STEPS" \
    --max_problems "$MAX_PROBLEMS" \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" \
    --out "$OUT_DIR/rollouts.shard${i}.jsonl" >>"$LOG" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[FATAL] generation failed" >&2; tail -40 "$LOG" >&2; exit 1; }

echo
echo "=== 3. results ==="
python scripts/analysis/online_bon_report.py \
  --rollouts "$OUT_DIR"/rollouts.shard*.jsonl \
  --out "$RUN_ROOT/online_bon_report.json" 2>&1 | tee -a "$LOG"

echo "[done] $OUT_DIR"
