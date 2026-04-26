#!/usr/bin/env bash
# Pull outputs from TamIA back to your laptop.
# Usage:
#   retrieve.sh              — sync $STORE/results/ and $STORE/probe_data/
#   retrieve.sh <JOBID>      — also fetch the job log for that job ID

set -euo pipefail

REMOTE_USER="${TAMIA_USER:-$USER}"
REMOTE_HOST="tamia.alliancecan.ca"
STORE="/project/aip-azouaq/${REMOTE_USER}"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

LOCAL_RESULTS="${REPO_ROOT}/results/tamia"
LOCAL_DATA="${REPO_ROOT}/data/tamia"

mkdir -p "$LOCAL_RESULTS" "$LOCAL_DATA"

echo "[retrieve] pulling results/ ..."
rsync -avz \
    "${REMOTE_USER}@${REMOTE_HOST}:${STORE}/results/" \
    "${LOCAL_RESULTS}/"

echo "[retrieve] pulling probe_data/ ..."
rsync -avz \
    "${REMOTE_USER}@${REMOTE_HOST}:${STORE}/probe_data/" \
    "${LOCAL_DATA}/"

if [[ -n "${1:-}" ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
    JOB_ID="$1"
    echo "[retrieve] fetching log for job $JOB_ID ..."
    scp "${REMOTE_USER}@${REMOTE_HOST}:~/CoT-checker/logs/*_${JOB_ID}.out" \
        "${LOCAL_RESULTS}/" 2>/dev/null || \
    scp "${REMOTE_USER}@${REMOTE_HOST}:~/CoT-checker/logs/*_${JOB_ID}.err" \
        "${LOCAL_RESULTS}/" 2>/dev/null || \
    echo "[retrieve] no log found for job $JOB_ID"
fi

echo "[retrieve] done"
echo "  results → ${LOCAL_RESULTS}/"
echo "  data    → ${LOCAL_DATA}/"
