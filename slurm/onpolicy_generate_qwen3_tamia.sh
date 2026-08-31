#!/bin/bash
#SBATCH --job-name=onpolicy_gen
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Stage 1 of the on-policy arm: Qwen3-8B-Base writes its own solutions to problems
# the off-policy grid was already scored on, and the gates decide whether the run
# is worth encoding and judging.
#
# The pilot (job 430576) ran 48 problems x 8 samples at ~1,700 trajectories/hour,
# which was ONE GPU: the generator does model.to(device) and had no sharding, so
# three H100s idled through it. It shards now, one process per GPU, which is where
# the 4x comes from. Shards stride the problem list so they finish together, and
# every PID's exit status is collected: a background cell that dies silently has
# already cost this project a job that reported COMPLETED having produced nothing.
#
# Sampling follows ReProbe for comparability: top-k 50, top-p 0.95, T=1.0. The
# pilot ran T=0.8 and landed at 0.435 trajectory accuracy; T=1.0 will sit a little
# lower, which the accuracy gate checks rather than assumes.
#
# Default size is the Stage 1 pilot (300 problems x 10). For the Stage 4 run set
# MAX_PROBLEMS=2000 and TIME to about 3h.
#
# NO INTERNET on compute nodes: weights must already be in $HF_CACHE.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/onpolicy_v1}"
PROBLEMS="${PROBLEMS:-$SCRATCH/cot_mech/dense_full_7b_v1/data/prm800k_test_2k.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
MAX_PROBLEMS="${MAX_PROBLEMS:-300}"
N_SAMPLES="${N_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
NUM_SHARDS="${NUM_SHARDS:-4}"
STEM="${STEM:-onpolicy_stage1}"
RUN_NAME="${RUN_NAME:-onpolicy_v1_stage1}"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
LOG_DIR="$RUN_ROOT/logs"; mkdir -p "$RUN_ROOT" "$LOG_DIR"
LOG_FILE="$LOG_DIR/onpolicy_gen-${SLURM_JOB_ID:-$$}.log"

SNAP="$HF_CACHE/hub/models--$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|--|g')/snapshots"
[[ -d "$SNAP" && -n "$(ls -A "$SNAP" 2>/dev/null)" ]] || {
  echo "[FATAL] no local snapshot for $MODEL_NAME_OR_PATH; download on the login node" >&2
  exit 2; }
[[ -f "$PROBLEMS" ]] || { echo "[FATAL] no problem file at $PROBLEMS" >&2; exit 2; }

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-onpolicy_gen}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH (base, offline)
problems   : $PROBLEMS  (first $MAX_PROBLEMS, PRM800K test)
sampling   : $N_SAMPLES per problem, T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K
shards     : $NUM_SHARDS on ${SLURM_GPUS_ON_NODE:-4} GPUs
out        : $RUN_ROOT/$STEM
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy

pids=(); tags=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  echo "[launch] shard $i -> GPU $i" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=$i python scripts/generate_onpolicy_steps.py \
    --fork_items "$PROBLEMS" --id_field problem_id \
    --out_dir "$RUN_ROOT" --stem "$STEM" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --model_dtype bfloat16 --run_name "$RUN_NAME" \
    --max_problems "$MAX_PROBLEMS" --n_samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE" --top_p "$TOP_P" --top_k "$TOP_K" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" --force >>"$LOG_FILE" 2>&1 &
  pids+=($!); tags+=("shard_$i")
done

fail=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[ok] ${tags[$j]}"; else echo "[FAIL] ${tags[$j]}"; fail=1; fi
done
if [[ "$fail" == "1" ]]; then tail -40 "$LOG_FILE" >&2; exit 1; fi

echo "=== Stage 1 gates ==="
set +e
python scripts/onpolicy/pilot_gates.py \
  --trajectories "$RUN_ROOT"/"$STEM".shard*_trajectories.jsonl \
  --reference_items "$PROBLEMS" \
  --tokenizer "$MODEL_NAME_OR_PATH" --local_files_only \
  --out "$RUN_ROOT/${STEM}_gates.json" | tee -a "$LOG_FILE"
gate_status=${PIPESTATUS[0]}
set -e

# The outcomes sidecar every T2 simulation reads. It needs no judge, so it is
# written here whether or not the gates passed: a NO-GO on reranking headroom is
# itself a number worth keeping.
python scripts/onpolicy/build_pb_traces.py \
  --trajectories "$RUN_ROOT"/"$STEM".shard*_trajectories.jsonl \
  --out_dir "$RUN_ROOT" --stem "$STEM" --force | tee -a "$LOG_FILE"

echo "[$(date)] onpolicy_gen done (gates exit $gate_status)"
exit $gate_status
