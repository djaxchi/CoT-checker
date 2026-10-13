#!/bin/bash
# Submit step_value_labels_v1 from the TamIA login node.
#   bash submit_step_value_labels.sh <snapshot_dir> [smoke|validate|full] [nodes]
# smoke:    trace_set_validate.json, 2 traces per shard, k=16, 20 min, into $R/smoke.
# validate: the 501 factorized_step_labels_v1 forks, cuts [s, s+1], k=16, one node,
#           into $R/validate. Gate (plan section 4) is checked by summarize_values.py
#           against the full run's track-1 shards before any label run.
# full:     trace_set_label.json, all cuts, k=16, on [nodes] nodes (default 4), into
#           $R/label. Walltime from the validate run's measured throughput (TIME=).
# The frozen set must already sit in $R/split; its sha256 is checked against its
# build audit here. Values passed through --export must not contain spaces or commas.
set -euo pipefail
SNAP="${1:?snapshot dir}"
MODE="${2:-validate}"
NODES="${3:-4}"
R=/project/aip-azouaq/$USER/cot_mech/step_value_labels_v1
case "$MODE" in
  smoke|validate) TAG=validate ;;
  full) TAG=label ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
python3 - "$R/split" "build_audit_$TAG.json" <<'PY'
import hashlib, json, sys, pathlib
d = pathlib.Path(sys.argv[1]); audit = json.loads((d / sys.argv[2]).read_text())
for name, want in audit["outputs"].items():
    got = hashlib.sha256((d / name).read_bytes()).hexdigest()
    if got != want:
        sys.exit(f"[FATAL] {name} sha256 {got} != {sys.argv[2]} {want}")
print("trace set checksums ok")
PY
OUT=$R/$MODE; [[ "$MODE" == full ]] && OUT=$R/label
if [[ "$MODE" != smoke && -n "$(ls "$OUT"/values.shard*.jsonl 2>/dev/null)" ]]; then
  echo "[FATAL] $OUT already has outputs (resume with RESUME=1 by hand)" >&2; exit 2
fi
mkdir -p "$OUT/logs"
JOBS="$OUT/submitted_jobs.txt"; echo "snapshot $SNAP mode $MODE" >> "$JOBS"
EXTRA=",SET_NAME=trace_set_$TAG.json,OUT=$OUT"
sb() { sbatch --parsable --output="$OUT/logs/%x-%j.out" --export=ALL,SNAP=$SNAP,R=$R$EXTRA "$@"; }
S="$SNAP/slurm/step_value_labels_tamia.sh"
case "$MODE" in
  smoke) j=$(EXTRA="$EXTRA,LIMIT=2" sb --time=00:20:00 --job-name=svl_smoke "$S"); echo "smoke $j" | tee -a "$JOBS" ;;
  validate) j=$(sb --time="${TIME:-01:00:00}" --job-name=svl_val "$S"); echo "validate $j" | tee -a "$JOBS" ;;
  full)
    for ((g = 0; g < NODES; g++)); do
      j=$(EXTRA="$EXTRA,GROUP=$g,NUM_SHARDS=$((4 * NODES))" sb --time="${TIME:?set TIME from the validate throughput}" \
          --job-name=svl_$g "$S")
      echo "label group $g $j" | tee -a "$JOBS"
    done ;;
esac
