#!/bin/bash
# Submit the 4-stage afterok DAG for ONE model size.
#
#   Stage A  encode PRM800K (4-way array)           [optional start dependency]
#   Stage B  merge + train probe        afterok: A
#   Stage C  evaluate ProcessBench (4-way array)    afterok: B
#   Stage D  aggregate + leaderboard    afterok: C
#
# Parallel WITHIN a model (A and C are 4-task arrays), sequential ACROSS models
# (the next model's Stage A is chained afterok on this model's Stage D).
#
# Usage:
#   submit_model_dag.sh <tag> [start_dep_jobid]
#
# Prints the Stage-D job id on stdout (so callers can chain the next model).
# A human-readable summary goes to stderr.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/models.env"

TAG="${1:?usage: submit_model_dag.sh <tag> [start_dep_jobid]}"
START_DEP="${2:-}"

if [[ -z "${S1MS_MODEL_ID[$TAG]:-}" ]]; then
  echo "[submit] unknown tag '$TAG'. Known: ${!S1MS_MODEL_ID[*]}" >&2
  exit 1
fi

# Smoke gate only for the 1.5B reproduction.
GATE=0
if [[ "$TAG" == "qwen2_5_1_5b" ]]; then GATE=1; fi

mkdir -p "$RUNS_ROOT/$TAG/logs"

# Pass the real (un-spooled) script dir so the stage jobs can find models.env /
# _common.sh; sbatch copies the script body to a spool dir where siblings are absent.
EXPORT_COMMON="ALL,TAG=$TAG,S1MS_DIR=$HERE"

dep_clause() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || echo ""; }

jidA=$(sbatch --parsable $(dep_clause "$START_DEP") \
        --export="$EXPORT_COMMON" "$HERE/stageA_encode_prm.sh")
jidB=$(sbatch --parsable --dependency=afterok:"$jidA" \
        --export="$EXPORT_COMMON" "$HERE/stageB_merge_train.sh")
jidC=$(sbatch --parsable --dependency=afterok:"$jidB" \
        --export="$EXPORT_COMMON" "$HERE/stageC_eval_pb.sh")
jidD=$(sbatch --parsable --dependency=afterok:"$jidC" \
        --export="$EXPORT_COMMON,GATE=$GATE" "$HERE/stageD_aggregate.sh")

{
  echo "[submit] $TAG (${S1MS_MODEL_ID[$TAG]})  gate=$GATE  start_dep=${START_DEP:-none}"
  echo "         A encode_prm  = $jidA  (array 0-3)"
  echo "         B merge_train = $jidB  (afterok:$jidA)"
  echo "         C eval_pb     = $jidC  (array 0-3, afterok:$jidB)"
  echo "         D aggregate   = $jidD  (afterok:$jidC)"
} >&2

echo "$jidD"
