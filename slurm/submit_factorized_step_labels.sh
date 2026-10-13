#!/bin/bash
# Submit factorized_step_labels_v1 from the TamIA login node.
#   bash submit_factorized_step_labels.sh <snapshot_dir> [pilot|smoke|full]
# pilot: 30 forks, m=4 k=8; {track1, track2} one H100:4 node each -> summary (CPU).
# smoke: the pilot split, 1 fork per shard and 20 reference traces, into $W/smoke.
# full:  fork_split_full.json, m=8 k=8; track 1 on 4 nodes (16 shards), track 2 on
#        one node -> summary into $R/full.
# The frozen split must already sit in $R/split (built and inspected before any
# GPU job); its sha256 is checked against its build audit here.
# Values passed through --export must not contain spaces or commas.
set -euo pipefail
SNAP="${1:?snapshot dir}"
MODE="${2:-pilot}"
R=/project/aip-azouaq/$USER/cot_mech/factorized_step_labels_v1
W=$SCRATCH/cot_mech/factorized_step_labels_v1
case "$MODE" in
  pilot|smoke) SPLIT_NAME=fork_split_v1.json; AUDIT=build_audit.json ;;
  full) SPLIT_NAME=fork_split_full.json; AUDIT=build_audit_full.json ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
python3 - "$R/split" "$AUDIT" <<'PY'
import hashlib, json, sys, pathlib
d = pathlib.Path(sys.argv[1]); audit = json.loads((d / sys.argv[2]).read_text())
for name, want in audit["outputs"].items():
    got = hashlib.sha256((d / name).read_bytes()).hexdigest()
    if got != want:
        sys.exit(f"[FATAL] {name} sha256 {got} != {sys.argv[2]} {want}")
print("split checksums ok")
PY
EXTRA=",SPLIT_NAME=$SPLIT_NAME"; T=""
if [[ "$MODE" == smoke ]]; then
  W=$W/smoke; EXTRA="$EXTRA,LIMIT=1,REF_LIMIT=20"; T="--time=00:20:00"
elif [[ "$MODE" == full ]]; then
  W=$W/full; EXTRA="$EXTRA,DEST_NAME=full"
fi
if [[ "$MODE" != smoke ]]; then
  [[ -e "$W/track1" || -e "$W/track2" ]] && { echo "[FATAL] $W already has outputs" >&2; exit 2; }
fi
mkdir -p "$W/logs"
JOBS="$W/submitted_jobs.txt"; echo "snapshot $SNAP mode $MODE" > "$JOBS"
sb() { sbatch --parsable --output="$W/logs/%x-%j.out" --export=ALL,SNAP=$SNAP,R=$R,W=$W$EXTRA "$@"; }
T1="$SNAP/slurm/factorized_step_labels_track1_tamia.sh"
T2="$SNAP/slurm/factorized_step_labels_track2_tamia.sh"
if [[ "$MODE" == full ]]; then
  t1=""
  for g in 0 1 2 3; do
    j=$(EXTRA="$EXTRA,M=8,K=8,GROUP=$g,NUM_SHARDS=16" sb --time=01:30:00 --job-name=fsl_t1_$g "$T1")
    t1="$t1:$j"; echo "track1 group $g $j" | tee -a "$JOBS"
  done
  t1=${t1#:}
else
  t1=$(sb $T "$T1"); echo "track1 $t1" | tee -a "$JOBS"
fi
t2=$(sb $T "$T2"); echo "track2 $t2" | tee -a "$JOBS"
if [[ "$MODE" != smoke ]]; then
  s=$(sb --dependency=afterok:$t1:$t2 "$SNAP/slurm/factorized_step_labels_summary_tamia.sh")
  echo "summary $s" | tee -a "$JOBS"
fi
