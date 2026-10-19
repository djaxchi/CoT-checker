#!/bin/bash
# risk_probe_v2 on TamIA (login node):
#   bash slurm/submit_risk_probe_v2.sh <snapshot_dir> smoke     # into $R2/smoke, 1 h
#   TIME=hh:mm:ss bash slurm/submit_risk_probe_v2.sh <snapshot_dir> full   # into $R2/prod
set -euo pipefail
SNAP="${1:?snapshot dir}"; MODE="${2:-smoke}"
R2=/project/aip-azouaq/$USER/cot_mech/risk_probe_v2
mkdir -p "$R2/slurm_out"
case "$MODE" in
  smoke) OUT=$R2/smoke; T=01:00:00; EX=",SMOKE=1" ;;
  full)  OUT=$R2/prod; T="${TIME:?set TIME from the smoke}"; EX="" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
# RESUME=1 continues an interrupted run in place (finished fits are skipped)
[[ -e "$OUT/fits" && "${RESUME:-0}" != 1 ]] && { echo "[FATAL] $OUT/fits exists; set RESUME=1 to continue it" >&2; exit 2; }
j=$(sbatch --parsable --time="$T" --job-name=rp2_$MODE --output="$R2/slurm_out/%x-%j.out" \
      --export=ALL,SNAP=$SNAP,OUT=$OUT$EX "$SNAP/slurm/risk_probe_v2_tamia.sh")
echo "$MODE $j" | tee -a "$R2/submitted_jobs.txt"
