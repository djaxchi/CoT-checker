#!/bin/bash
# Phase 2 of the sweep: submit 3B -> 7B -> 14B -> 32B, sequentially chained.
#
# Each model's Stage A is afterok-chained on the PREVIOUS model's Stage D, so a
# model only starts once the smaller one has produced its leaderboard row. The
# first model (3B) is chained on the 1.5B smoke gate job id passed as $1, so the
# larger sizes never run unless 1.5B reproduced Sprint 1.
#
# Usage:
#   ./launch_rest.sh <smoke_aggregate_jobid>     # normal: gate on 1.5B success
#   FORCE=1 ./launch_rest.sh                      # start 3B immediately (no gate)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE_DEP="${1:-}"
if [[ -z "$SMOKE_DEP" && "${FORCE:-0}" != "1" ]]; then
  echo "usage: launch_rest.sh <smoke_aggregate_jobid>" >&2
  echo "       (or FORCE=1 launch_rest.sh to start 3B with no 1.5B gate)" >&2
  exit 1
fi

REST=(qwen2_5_3b qwen2_5_7b qwen2_5_14b qwen2_5_32b)

prev="$SMOKE_DEP"
for tag in "${REST[@]}"; do
  prev="$("$HERE/submit_model_dag.sh" "$tag" "$prev")"
done

echo "[launch_rest] submitted ${REST[*]} chained; final aggregate job = $prev" >&2
echo "$prev"
