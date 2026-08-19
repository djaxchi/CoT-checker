#!/bin/bash
# Add the concat[pre-step boundary, step_stats] row to the 7B leaderboard.
#
# Why this row: in the lookahead ceiling test, handing the probe the pre-step
# boundary state as its OWN vector alongside the pooled step (pc) beat the pooled
# step alone (cur) on 3 of 4 ProcessBench subsets, by +0.162 AUROC on omnimath.
# That is not extra information (every state is already causally contextualized
# by the question and prior steps) but extra linear accessibility: the probe can
# weigh the step against where it started. No leaderboard row has this form.
# last_token/step_stats read the step span with no anchor, and step_delta uses
# the boundary but FORCES S_t - S_{t-1}, discarding the absolute level. Concat is
# strictly more expressive than the subtraction (tests/repstore/test_derive_delta
# .py::test_boundary_stats_is_more_expressive_than_delta), so the prediction is
# that it beats both last_token (0.394) and step_delta (0.361) F1_PB @ calib-20.
#
# Stage 1 derives the vectors offline from the existing token store (no GPU, no
# re-encoding); stage 2 trains and evaluates, and only starts if stage 1 succeeds.
#
# Run from the TamIA login node:  bash slurm/submit_boundary_stats_row.sh

set -euo pipefail
cd "${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/dense_full_7b_v1}"
TAG=boundary_stats

d=$(sbatch --parsable --job-name=derive_${TAG} \
      --export=ALL,READOUT=$TAG,TAG=$TAG \
      slurm/derive_rep_7b_tamia.sh)
echo "derive  -> job $d"

h=$(sbatch --parsable --dependency=afterok:$d --job-name=harness_${TAG} \
      --export=ALL,REP_TAG=$TAG,RUN_TAG=$TAG,METHOD=dense_linear,PB_ROOT=$RUN_ROOT/cache/qwen2_5_7b_${TAG}_pb \
      slurm/train_harness_7b_tamia.sh)
echo "harness -> job $h  (afterok:$d)"
echo
echo "when done, calib-20 the scores:"
echo "  python scripts/analysis/pb_threshold_calibration.py --calib_sizes 20 \\"
echo "    --scores $RUN_ROOT/runs/${TAG}_dense_linear/pb_step_scores_*.jsonl"
