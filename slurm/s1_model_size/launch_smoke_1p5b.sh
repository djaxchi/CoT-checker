#!/bin/bash
# Phase 1 of the sweep: submit ONLY the 1.5B smoke reproduction DAG.
#
# The 1.5B Stage D enforces the known Sprint 1 DenseLinear result
# (val macro F1_PB ~= 0.1855, oracle ~= 0.3773) within tolerance and exits
# non-zero otherwise, so a failed reproduction blocks everything chained after
# it. Once this DAG's aggregate job succeeds, launch the rest with:
#
#     ./launch_rest.sh <printed_aggregate_jobid>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

jidD="$("$HERE/submit_model_dag.sh" qwen2_5_1_5b)"

cat >&2 <<EOF

================================================================
1.5B smoke DAG submitted. Aggregate (gate) job id = $jidD
After it completes successfully, submit the rest with:

    $HERE/launch_rest.sh $jidD

If the gate job FAILS, the 1.5B run did not reproduce Sprint 1.
Stop and debug; do NOT launch the larger models.
================================================================
EOF

echo "$jidD"
