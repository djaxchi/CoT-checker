#!/bin/bash
# Fan the within-PB lookahead ceiling test out to one job per ProcessBench subset.
#
# The unsharded job was killed by walltime four times (27/55/102/51 min) and never
# reached olympiadbench or omnimath. Timing evidence from job 389107: gsm8k + math
# with windows "1 -1" took ~1h41 together. Per-subset walltimes below are sized
# from that, with the two long-solution subsets given the most room.
#
# Run from the TamIA login node:  bash slurm/submit_lookahead_shards.sh

set -euo pipefail
cd "${PROJECT_ROOT:-$HOME/CoT-checker}"

submit () {   # $1 = subset, $2 = walltime
  local jid
  jid=$(sbatch --parsable --time="$2" --job-name="lookahead_$1" \
        --export=ALL,SUBSETS="$1" slurm/analyze_lookahead_cpu_tamia.sh)
  echo "$1 -> job $jid (time $2)"
}

submit gsm8k         02:00:00
submit math          04:00:00
submit olympiadbench 08:00:00
submit omnimath      08:00:00
