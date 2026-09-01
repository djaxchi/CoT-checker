#!/bin/bash
#SBATCH --job-name=onpolicy_transfer
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# Stages 3 and 4: encode the on-policy traces at layer 35 under BOTH contexts,
# then point the already-trained grid at them and ask whether the ranking holds.
#
# No cell is retrained. The nineteen dense cells are on disk with their weights,
# and evaluate_processbench takes any store split whose meta carries id /
# step_idx / label / n_steps, so the same nineteen verifiers are pointed at
# on-policy text and the text distribution is the only thing that changed.
#
# The two contexts are two different experiments.
#
#   verifier    the template the whole off-policy grid was encoded under. Reading
#               on-policy text through it changes exactly one thing against the
#               off-policy arm, the text distribution. This is the controlled
#               comparison the rank claim needs, and it is the primary arm.
#   generation  the context the sampler actually ran under, rebuilt from the
#               model's own prefix. A forward pass over that string reproduces the
#               generative states of the step's tokens. This is "on-policy states"
#               in the strict sense, and it is the second perturbation.
#
# They are encoded as two subsets of one store, so a cell is scored on each by
# pointing --split_dir at one or the other and nothing else changes.
#
# Input is the ProcessBench-shaped traces file from scripts/onpolicy/build_pb_traces.py,
# so the meta rows carry id / step_idx / label / n_steps and evaluate_processbench
# needs no modification at all.
#
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/onpolicy_v1}"
# The unlabelled set: every gradeable trajectory, placeholder label. The
# downstream simulations score against the trajectory's outcome and never read a
# step label, so this arm needs no judge and no rollouts and costs nothing.
TRACES="${TRACES:-$RUN_ROOT/onpolicy_stage1_unlabelled.jsonl}"
STEM="${STEM:-onpolicy_stage1}"
REP_ROOT="${REP_ROOT:-$RUN_ROOT/repstore/onpolicy_step_spans}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
STYLES="${STYLES:-verifier generation}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
# Index 35 is resid_post of block 34, the last genuine resid_post and the layer
# the whole Qwen3 grid reads. Index -1 is post-final-RMSNorm and is NOT the same.
LAYER="${LAYER:-35}"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
LOG_DIR="$RUN_ROOT/logs"; mkdir -p "$REP_ROOT" "$LOG_DIR"
LOG_FILE="$LOG_DIR/onpolicy_encode-${SLURM_JOB_ID:-$$}.log"

SNAP="$HF_CACHE/hub/models--$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|--|g')/snapshots"
[[ -d "$SNAP" && -n "$(ls -A "$SNAP" 2>/dev/null)" ]] || {
  echo "[FATAL] no local snapshot for $MODEL_NAME_OR_PATH; download on the login node" >&2
  exit 2; }
if [[ ! -f "$TRACES" ]]; then
  echo "[build] $TRACES missing; building the unlabelled set from the trajectories"
  python scripts/onpolicy/build_pb_traces.py \
    --trajectories "$RUN_ROOT"/"$STEM".shard*_trajectories.jsonl \
    --out_dir "$RUN_ROOT" --stem "$STEM" \
    --unlabelled "$TRACES" --force
fi
[[ -f "$TRACES" ]] || { echo "[FATAL] could not build $TRACES" >&2; exit 2; }

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-onpolicy_encode}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH (offline)
traces     : $TRACES  ($(wc -l <"$TRACES") traces)
rep_root   : $REP_ROOT  (span-only, layer $LAYER)
styles     : $STYLES
shards     : $NUM_SHARDS on ${SLURM_GPUS_ON_NODE:-4} GPUs   batch: $BATCH_SIZE
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

for style in $STYLES; do
  echo "[style] $style" | tee -a "$LOG_FILE"
  pids=(); tags=()
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/encode_processbench_token_store.py \
      --raw_specs "${style}:${TRACES}" \
      --rep_root "$REP_ROOT" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
      --span_only --prompt_style "$style" \
      --layer "$LAYER" --max_seq_len "$MAX_SEQ_LEN" \
      --batch_size "$BATCH_SIZE" --model_dtype "${MODEL_DTYPE:-bfloat16}" \
      --shard_idx "$i" --num_shards "$NUM_SHARDS" >>"$LOG_FILE" 2>&1 &
    pids+=($!); tags+=("${style}_shard_$i")
  done
  fail=0
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "[ok] ${tags[$j]}"; else echo "[FAIL] ${tags[$j]}"; fail=1; fi
  done
  [[ "$fail" == "0" ]] || { tail -40 "$LOG_FILE" >&2; exit 1; }
done

python - <<PY | tee -a "$LOG_FILE"
from pathlib import Path
import sys
sys.path.insert(0, "$PROJECT_ROOT")
from src.repstore import split_fingerprint
from src.repstore.store import ShardedRepSplit
for style in "$STYLES".split():
    d = Path("$REP_ROOT") / style
    v = ShardedRepSplit(d)
    print(f"{style:<11} {len(v):>7,} steps  dim {v.spec.dim}  "
          f"layer {v.spec.layer}  prompt_style {v.spec.prompt_style}  "
          f"fingerprint {split_fingerprint(d)}")
PY

echo
echo "=== score the trained cells on each context ==="
GRID="${GRID:-$SCRATCH/cot_mech/qwen3_8b_v1/runs/rep_grid_q3}"
PRM_STORE="${PRM_STORE:-$SCRATCH/cot_mech/qwen3_8b_v1/repstore/step_spans}"
VEC_CACHE="${VEC_CACHE:-$SCRATCH/cot_mech/qwen3_8b_v1/cache}"
# rep_grid_q3 predates protocol.rescale; the setting is stated rather than
# inferred, and the scorer refuses to guess it.
ASSUME_RESCALE="${ASSUME_RESCALE:-none}"

for style in $STYLES; do
  python scripts/onpolicy/score_cells_on_split.py \
    --cells "$GRID" --glob "*__seed4*" \
    --split_dir "$REP_ROOT/$style" --split_name "onpolicy_$style" \
    --prm_store "$PRM_STORE" --vec_cache_dir "$VEC_CACHE" \
    --stats_cache_dir "$RUN_ROOT/rescale_stats" \
    --assume_rescale "$ASSUME_RESCALE" \
    --summary "$RUN_ROOT/scores_onpolicy_${style}.json" 2>&1 | tee -a "$LOG_FILE"
done

echo
echo "=== the off-policy x-axis, recomputed under the leaderboard protocol ==="
OFFPOLICY_METRIC="${OFFPOLICY_METRIC:-$RUN_ROOT/offpolicy_f1pb.json}"
[[ -f "$OFFPOLICY_METRIC" ]] || python scripts/onpolicy/export_offpolicy_metric.py \
  --grid_root "$GRID" --out "$OFFPOLICY_METRIC" 2>&1 | tee -a "$LOG_FILE"

echo
echo "=== T2: does the benchmark rank predict downstream usefulness? ==="
# No step labels anywhere in this block. Every number is scored against whether
# the solution reached the right answer, which the grader already settled.
for style in $STYLES; do
  echo "--- $style ---"
  python scripts/analysis/onpolicy_downstream.py \
    --grid_root "$GRID" --scores_name "onpolicy_$style" \
    --outcomes "$RUN_ROOT/${STEM}_outcomes.jsonl" \
    --offpolicy_metric "$OFFPOLICY_METRIC" \
    --out "$RUN_ROOT/downstream_${style}.json" 2>&1 | tee -a "$LOG_FILE"
done

# T1's strict form needs first-error labels and runs separately once a judged or
# rollout-labelled split exists; the rank script refuses the unlabelled one.
if [[ -f "$RUN_ROOT/${STEM}_rollout_pb_traces.jsonl" ]]; then
  echo
  echo "=== T1: does the ranking survive, on the labelled split? ==="
  for style in $STYLES; do
    python scripts/analysis/onpolicy_rank_transfer.py \
      --grid_root "$GRID" --onpolicy_name "onpolicy_labelled_$style" \
      --length_meta_on "$REP_ROOT/$style" \
      --length_meta_off "$SCRATCH/cot_mech/qwen3_8b_v1/repstore/pb_step_spans/gsm8k" \
      --out "$RUN_ROOT/rank_transfer_${style}.json" 2>&1 | tee -a "$LOG_FILE" || true
  done
fi

du -sh "$REP_ROOT"/* 2>/dev/null | tee -a "$LOG_FILE"
echo "[$(date)] onpolicy_transfer done"
