#!/bin/bash
# risk_probe_v1 on TamIA (login node), after build_views.py froze $R/views:
#   bash slurm/submit_risk_probe_v1.sh <snapshot_dir> smoke
#       encode the whole manifest, 5 tiny validity fits, rescoring and row dump,
#       into $R/smoke (about 30 min)
#   [AFTER=jobid:jobid] bash slurm/submit_risk_probe_v1.sh <snapshot_dir> full
#       the encode/fit job into $R/prod, then the readouts job (afterok on it and
#       on AFTER, e.g. the label jobs) into $R/prod/readouts and $R/analysis
#   [AFTER=...] bash slurm/submit_risk_probe_v1.sh <snapshot_dir> readouts <run_dir> <tag>
#       readouts only, on an existing encode/fit run (e.g. the smoke, tag smoke)
#   bash slurm/submit_risk_probe_v1.sh <snapshot_dir> pb
#       post hoc: encode the ProcessBench `full` step views ($R/pb_views, built here by
#       build_pb_manifest.py), score them with the validity probes and the readouts,
#       into $R/pb (no probe fit; FITS_FILE is empty)
# EXTRA_STEPS holds a space, so it travels in the exported environment.
set -euo pipefail
SNAP="${1:?snapshot dir}"; MODE="${2:-smoke}"
R=/project/aip-azouaq/$USER/cot_mech/risk_probe_v1
[[ -s "$R/views/encode_manifest.jsonl" ]] || { echo "[FATAL] build the view manifest first" >&2; exit 2; }
mkdir -p "$R/slurm_out"
export EXTRA_STEPS="bash $SNAP/slurm/risk_probe_v1_extra.sh"
enc() {  # $1 out root, $2 fits file, $3 time, $4 name; MANIFEST_DIR overrides the manifest
  sbatch --parsable --time="$3" --job-name="$4" --output="$R/slurm_out/%x-%j.out" \
    --export=ALL,SNAP=$SNAP,MANIFEST=${MANIFEST_DIR:-$R/views},OUT_ROOT=$1,FITS_FILE=$2,NUM_SHARDS=16 \
    "$SNAP/slurm/bidirectional_token_probe_train_tamia.sh"
}
read_job() {  # $1 run dir, $2 tag, $3 dependency list
  local dep=(); [[ -n "$3" ]] && dep=(--dependency="afterok:$3")
  sbatch --parsable "${dep[@]}" --time="${READ_TIME:-02:00:00}" --job-name=rp1_read_$2 \
    --output="$R/slurm_out/%x-%j.out" \
    --export=ALL,SNAP=$SNAP,RUN=$1,FIT=$1/readouts,ANALYSIS=$R/analysis_$2,N_PERM=${N_PERM:-200}${STEP_VALUES:+,STEP_VALUES=$STEP_VALUES} \
    "$SNAP/slurm/risk_probe_v1_readouts_tamia.sh"
}
case "$MODE" in
  smoke)
    [[ -e "$R/smoke/fits" ]] && { echo "[FATAL] $R/smoke/fits exists" >&2; exit 2; }
    j=$(enc "$R/smoke" "$SNAP/experiments/risk_probe_v1/xfit5_smoke.fits" 00:45:00 rp1_smoke)
    echo "smoke $j" | tee -a "$R/submitted_jobs.txt" ;;
  full)
    [[ -e "$R/prod/fits" ]] && { echo "[FATAL] $R/prod/fits exists; never submit twice into it" >&2; exit 2; }
    j=$(enc "$R/prod" "$SNAP/experiments/risk_probe_v1/xfit5.fits" "${TIME:-01:30:00}" rp1)
    echo "encode/fit $j" | tee -a "$R/submitted_jobs.txt"
    k=$(read_job "$R/prod" prod "$j${AFTER:+:$AFTER}")
    echo "readouts $k (afterok $j${AFTER:+:$AFTER})" | tee -a "$R/submitted_jobs.txt" ;;
  pb)
    [[ -e "$R/pb" ]] && { echo "[FATAL] $R/pb exists" >&2; exit 2; }
    [[ -s "$R/pb_views/encode_manifest.jsonl" ]] || python3 "$SNAP/scripts/risk_probe_v1/build_pb_manifest.py" \
      --src /project/aip-azouaq/$USER/cot_mech/context_ablation_v3/views/full --out "$R/pb_views"
    export EXTRA_STEPS="bash $SNAP/slurm/risk_probe_v1_pb_extra.sh"
    export SHARDS_TO_ENCODE="$(python3 -c "import json;print(' '.join(map(str,json.load(open('$R/pb_views/build_audit.json'))['shards'])))")"
    j=$(MANIFEST_DIR=$R/pb_views enc "$R/pb" /dev/null "${TIME:-01:00:00}" rp1_pb)
    echo "pb $j" | tee -a "$R/submitted_jobs.txt" ;;
  readouts)
    k=$(read_job "${3:?run dir}" "${4:?tag}" "${AFTER:-}")
    echo "readouts $k on $3" | tee -a "$R/submitted_jobs.txt" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
