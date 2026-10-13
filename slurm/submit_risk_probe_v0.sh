#!/bin/bash
# risk_probe_v0 on TamIA (login node): bash slurm/submit_risk_probe_v0.sh <snapshot_dir>
# One H100:4 job: encode the frozen view manifest (L35, node-local), 3 xfit fits,
# then slurm/risk_probe_v0_extra.sh (rescoring of the §21.18 checkpoints, pooled
# features). EXTRA_STEPS holds a space, so it travels in the exported environment
# rather than in --export.
set -euo pipefail
SNAP="${1:?snapshot dir}"
R=/project/aip-azouaq/$USER/cot_mech/risk_probe_v0
[[ -s "$R/views/encode_manifest.jsonl" ]] || { echo "[FATAL] build the view manifest first" >&2; exit 2; }
[[ -e "$R/prod/fits" ]] && { echo "[FATAL] $R/prod/fits exists; never submit twice into it" >&2; exit 2; }
mkdir -p "$R/slurm_out"
export EXTRA_STEPS="bash $SNAP/slurm/risk_probe_v0_extra.sh"
j=$(sbatch --parsable --time=01:30:00 --job-name=rp0 --output="$R/slurm_out/%x-%j.out" \
  --export=ALL,SNAP=$SNAP,MANIFEST=$R/views,OUT_ROOT=$R/prod,FITS_FILE=$SNAP/experiments/risk_probe_v0/xfit.fits,NUM_SHARDS=16 \
  "$SNAP/slurm/bidirectional_token_probe_train_tamia.sh")
echo "risk_probe_v0 job $j" | tee "$R/submitted_jobs.txt"
