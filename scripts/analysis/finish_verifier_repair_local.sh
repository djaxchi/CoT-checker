#!/bin/bash
# Wait for the known extraction job, retrieve its outputs, then score locally.
set -euo pipefail
JOB_ID="${1:?Pass the submitted repair extraction job ID}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid job ID' >&2; exit 2; }
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"
REMOTE=dchikhi@tamia.alliancecan.ca
REMOTE_RUN=/project/aip-azouaq/dchikhi/cot_mech/verifier_repair_v1/run_01
LOCAL_RUN=results/verifier_repair_v1
mkdir -p "$LOCAL_RUN"
[[ ! -e "$LOCAL_RUN/scores" && ! -e "$LOCAL_RUN/analysis" ]] || {
  echo 'Scoring/analysis output already exists; inspect before retrying.' >&2; exit 2;
}
START_SECONDS=$SECONDS
while (( SECONDS - START_SECONDS < 21600 )); do
  STATE="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE" "sacct -X -n -j $JOB_ID --format=State -P" | head -n 1 | tr -d '[:space:]')"
  printf '%s job=%s state=%s\n' "$(date -u +%FT%TZ)" "$JOB_ID" "$STATE"
  case "$STATE" in
    COMPLETED) break ;;
    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*)
      echo 'Extraction did not complete successfully.' >&2; exit 1 ;;
  esac
  sleep 60
done
[[ "$STATE" == COMPLETED ]] || { echo 'Six-hour wait limit reached.' >&2; exit 1; }
mkdir -p "$LOCAL_RUN/extraction"
rsync -a --partial "$REMOTE:$REMOTE_RUN/" "$LOCAL_RUN/extraction/"
.venv/bin/python scripts/analysis/verifier_repair_experiment.py score \
  --extraction "$LOCAL_RUN/extraction" --out "$LOCAL_RUN/scores"
.venv/bin/python scripts/analysis/verifier_repair_analysis.py \
  --extraction "$LOCAL_RUN/extraction" --scores "$LOCAL_RUN/scores" --out "$LOCAL_RUN/analysis"
date -u +%FT%TZ > "$LOCAL_RUN/local_completed_at.txt"
echo "Complete: $PROJECT_DIR/$LOCAL_RUN/analysis/summary.md"
