#!/bin/bash
# Submit the instruct_arm_v1 DAG. Run this on the TamIA login node, not as a job.
#
# Three stages chained with afterok. The two encodes are independent of each
# other, so they go in together and the trainer waits on both; if two nodes are
# free the wall clock halves.
#
# Everything except the backbone is held at the Base arm's settings, because the
# backbone is the only variable this comparison can attribute a difference to.
# Design and the reference numbers to beat are in docs/instruct_arm_v1_plan.md.
#
# The Base reference cell (qwen3_8b_v1/runs/rep_grid_q3, job 429667, 2026-08-26)
# predates --rescale (983c890, 2026-08-28), so it trained on raw states. The grid
# script now defaults to RESCALE=zscore; it is pinned to none here, or the
# comparison would change the input scaling along with the backbone.
#
# Walltimes are the Base arm's measured ones with margin, not the plan's guesses:
# PRM800K encode 52m (429618), ProcessBench encode 6m (429619), the d512 cell
# 2h01m for seed 42 with its search and 1h21m per reused seed.
#
# PRECONDITION, checked below rather than assumed: the weights must already be in
# HF_CACHE, since compute nodes have no internet.
#   HF_HOME=/project/aip-azouaq/$USER/hf_cache hf download Qwen/Qwen3-8B

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_8b_instruct_v1}"
# The split JSONs the Base arm froze. Reused unchanged: a new split here would
# make the two arms incomparable for a reason that has nothing to do with the
# backbone.
DATA_DIR="${DATA_DIR:-$SCRATCH/cot_mech/dense_full_7b_v1/data}"
LAYER="${LAYER:-35}"          # hidden-states index for block 34 of 36
CELLS="${CELLS:-step_tokens transformer:d512,l2,f2048,h8}"
SEEDS="${SEEDS:-42 43 44}"
OUT_ROOT="${OUT_ROOT:-$RUN_ROOT/runs/rep_grid_q3}"
# The Base cells ran with a 300 GB per-cell preload budget and the 163 GB train
# split preloaded. At most two cells run at once here (seed 42 alone, then 43 and
# 44 together), so 200 GB each fits the 500 GB node.
PRELOAD_BUDGET_GB="${PRELOAD_BUDGET_GB:-200}"

snap="$HF_CACHE/hub/models--${MODEL/\//--}/snapshots"
[[ -d "$snap" && -n "$(ls -A "$snap" 2>/dev/null)" ]] || {
  echo "[FATAL] $MODEL is not in $HF_CACHE." >&2
  echo "        HF_HOME=$HF_CACHE hf download $MODEL" >&2
  exit 2; }
n_shards=$(find "$snap"/ -name "*.safetensors" | wc -l)
[[ "$n_shards" -ge 1 ]] || { echo "[FATAL] no safetensors under $snap" >&2; exit 2; }
echo "[ok] weights present: $snap ($n_shards safetensors)"

common_env="MODEL_NAME_OR_PATH=$MODEL,HF_CACHE=$HF_CACHE,RUN_ROOT=$RUN_ROOT,DATA_DIR=$DATA_DIR,LAYER=$LAYER"

echo "=== 1. encode PRM800K step spans ==="
J1=$(sbatch --parsable --time=02:30:00 --job-name=prm_spanstore_instruct \
      --export=ALL,"$common_env",REP_ROOT="$RUN_ROOT/repstore/step_spans" \
      slurm/encode_prm800k_span_store_qwen3_tamia.sh)
echo "  job $J1"

echo "=== 2. encode ProcessBench step spans ==="
J2=$(sbatch --parsable --time=01:00:00 --job-name=pb_spanstore_instruct \
      --export=ALL,"$common_env",REP_ROOT="$RUN_ROOT/repstore/pb_step_spans" \
      slurm/encode_processbench_span_store_qwen3_tamia.sh)
echo "  job $J2"

echo "=== 3. train transformer L, $SEEDS, after both encodes ==="
J3=$(sbatch --parsable --time=05:30:00 --dependency=afterok:"$J1":"$J2" \
      --job-name=train_instruct_L \
      --export=ALL,RUN_ROOT="$RUN_ROOT",PRM_STORE="$RUN_ROOT/repstore/step_spans",PB_STORE="$RUN_ROOT/repstore/pb_step_spans",OUT_ROOT="$OUT_ROOT",VEC_CACHE="$RUN_ROOT/cache/grid_vectors",CELLS="$CELLS",SEEDS="$SEEDS",RESCALE=none,PRELOAD_BUDGET_GB="$PRELOAD_BUDGET_GB" \
      slurm/train_rep_grid_7b_tamia.sh)
echo "  job $J3"

cat <<EOF

Submitted: encode $J1, $J2 -> train $J3

Read this first, before anything else:
  $OUT_ROOT/step_tokens__transformer_d512_l2_f2048_h8__seed42/results.json

Against the Base arm's transformer L, seed 42:
  in-domain AUROC          0.8978
  ProcessBench val-sel avg 0.427
  ProcessBench oracle avg  0.576

AUROC is the comparison. The three thresholds are reported alongside it because
the gap between them is a calibration result, not a ranking one.

The artifact audit and the Instruct TTS pool are not chained here: the audit
needs the trained probe to measure weight localisation.
EOF
