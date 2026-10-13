#!/bin/bash
# factorized_step_labels_v1 pilot on Rorqual, a hedge mirroring the TamIA DAG
# (login node): bash submit_factorized_step_labels_rorqual.sh <snapshot_dir>
#   4 t1 shards | t2ref -> 4 t2 shards; the summary is a seconds-long CPU step run after
# Expects the frozen split in $R/split; checks it against build_audit.json.
set -euo pipefail
SNAP="${1:?snapshot dir}"
ACCOUNT="${ACCOUNT:-def-azouaq}"
R="${R:-$SCRATCH/cot_mech/factorized_step_labels_v1}"
python3 - "$R/split" <<'PY'
import hashlib, json, sys, pathlib
d = pathlib.Path(sys.argv[1]); audit = json.loads((d / "build_audit.json").read_text())
for name, want in audit["outputs"].items():
    got = hashlib.sha256((d / name).read_bytes()).hexdigest()
    if got != want:
        sys.exit(f"[FATAL] {name} sha256 {got} != build_audit {want}")
print("split checksums ok")
PY
[[ -e "$R/track1" || -e "$R/track2" ]] && { echo "[FATAL] $R already has outputs" >&2; exit 2; }
mkdir -p "$R/logs"
JOBS="$R/submitted_jobs_rorqual.txt"; echo "snapshot $SNAP" > "$JOBS"
S="$SNAP/slurm/factorized_step_labels_shard_rorqual.sh"
sb() { sbatch --parsable --account="$ACCOUNT" --output="$R/logs/%x-%j.out" "$@"; }
all=""
for s in 0 1 2 3; do
  j=$(sb --job-name=fsl_t1_$s --export=ALL,SNAP=$SNAP,R=$R,TASK=t1,SHARD=$s "$S"); all="$all:$j"
  echo "t1 shard $s $j" | tee -a "$JOBS"
done
ref=$(sb --time=00:30:00 --job-name=fsl_t2ref --export=ALL,SNAP=$SNAP,R=$R,TASK=t2ref "$S")
echo "t2ref $ref" | tee -a "$JOBS"
for s in 0 1 2 3; do
  j=$(sb --time=00:30:00 --dependency=afterok:$ref --job-name=fsl_t2_$s \
        --export=ALL,SNAP=$SNAP,R=$R,TASK=t2,SHARD=$s "$S"); all="$all:$j"
  echo "t2 shard $s $j" | tee -a "$JOBS"
done
echo "all ${all#:}; summarize afterwards with summarize_pilot.py (CPU, seconds)" | tee -a "$JOBS"
