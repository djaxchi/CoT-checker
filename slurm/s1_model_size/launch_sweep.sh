#!/bin/bash
# Convenience: submit the WHOLE sequential sweep in one command.
#
#   1.5B (smoke gate) -> 3B -> 7B -> 14B -> 32B
#
# Every model is afterok-chained on the previous model's Stage D, so:
#   * models run strictly one size at a time (parallel only within a size),
#   * a failed 1.5B gate (or any failed stage) stops everything downstream,
#   * completed smaller-model outputs are never deleted or rerun.
#
# Prefer the two-phase launch_smoke_1p5b.sh + launch_rest.sh when you want to
# eyeball the 1.5B reproduction before committing the big GPUs. This script is
# the same DAG, submitted up front.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/models.env"

prev=""
for tag in "${S1MS_TAGS[@]}"; do
  prev="$("$HERE/submit_model_dag.sh" "$tag" "$prev")"
done

echo "[launch_sweep] full sweep submitted (${S1MS_TAGS[*]}); final job = $prev" >&2
echo "$prev"
