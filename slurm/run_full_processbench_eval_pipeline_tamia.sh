#!/bin/bash
# One-shot launcher: submits the three stages of the full-ProcessBench
# evaluation pipeline with sbatch dependencies. Stage outputs land under
# $SCRATCH/cot_mech/prestudy_v1/.
#
# Stages:
#   A) encode_processbench_full_prestudy_tamia.sh      (dense PB cache)
#   B) extract_processbench_full_ssae_tamia.sh          (SSAE PB latents)
#   C) evaluate_existing_probes_full_processbench_tamia.sh   (leaderboards)
#
# Stages B and C are submitted with --dependency=afterok on the previous
# stage. C depends on both A and B.
#
# Skip stages you've already done by setting:
#   SKIP_ENCODE=1     # skip A; B and C will not wait on A
#   SKIP_EXTRACT=1    # skip B; C will not wait on B
#
# Forward FORCE / SKIP_MISSING / METHODS as needed:
#   sbatch_pipeline.sh   (no sbatch flags here; this is a plain shell driver)
#
# Usage:
#   bash slurm/run_full_processbench_eval_pipeline_tamia.sh

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"

A_SCRIPT="$SLURM_DIR/encode_processbench_full_prestudy_tamia.sh"
B_SCRIPT="$SLURM_DIR/extract_processbench_full_ssae_tamia.sh"
C_SCRIPT="$SLURM_DIR/evaluate_existing_probes_full_processbench_tamia.sh"

dep_chain=""
A_JID=""
if [[ "${SKIP_ENCODE:-0}" != "1" ]]; then
  A_JID=$(sbatch --parsable "$A_SCRIPT")
  echo "[pipeline] submitted A (dense PB encode): job_id=$A_JID"
  dep_chain="afterok:$A_JID"
else
  echo "[pipeline] SKIP_ENCODE=1 -> not submitting A"
fi

B_JID=""
if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
  if [[ -n "$dep_chain" ]]; then
    B_JID=$(sbatch --parsable --dependency="$dep_chain" "$B_SCRIPT")
  else
    B_JID=$(sbatch --parsable "$B_SCRIPT")
  fi
  echo "[pipeline] submitted B (SSAE PB extract): job_id=$B_JID"
fi

dep_for_C=""
parts=()
[[ -n "$A_JID" ]] && parts+=("afterok:$A_JID")
[[ -n "$B_JID" ]] && parts+=("afterok:$B_JID")
if [[ ${#parts[@]} -gt 0 ]]; then
  dep_for_C="$(IFS=, ; echo "${parts[*]}")"
fi

if [[ -n "$dep_for_C" ]]; then
  C_JID=$(sbatch --parsable --dependency="$dep_for_C" "$C_SCRIPT")
else
  C_JID=$(sbatch --parsable "$C_SCRIPT")
fi
echo "[pipeline] submitted C (full PB eval): job_id=$C_JID"

cat <<EOF
[pipeline] queue summary:
   A (encode) : ${A_JID:-skipped}
   B (extract): ${B_JID:-skipped}
   C (eval)   : $C_JID

Monitor:
   squeue -u \$USER -o "%.10i %.20j %.2t %.10M %R"
   tail -F slurm-${A_JID:-N}*.out slurm-${B_JID:-N}*.out slurm-${C_JID}*.out
EOF
