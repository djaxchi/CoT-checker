#!/usr/bin/env bash
# Monitor TamIA jobs from your laptop.
# Usage:
#   monitor.sh              — show all your running/pending jobs
#   monitor.sh <JOBID>      — tail the live log for a specific job
#   monitor.sh --history    — show recent job history with exit codes

set -euo pipefail

REMOTE_USER="${TAMIA_USER:-$USER}"
REMOTE_HOST="tamia.alliancecan.ca"
REMOTE_DIR="~/CoT-checker"

case "${1:-}" in
    --history)
        ssh "${REMOTE_USER}@${REMOTE_HOST}" \
            "sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode,NodeList | tail -30"
        ;;
    "")
        ssh "${REMOTE_USER}@${REMOTE_HOST}" "squeue -u $USER"
        ;;
    [0-9]*)
        JOB_ID="$1"
        echo "[monitor] tailing log for job $JOB_ID — Ctrl-C to stop"
        ssh -t "${REMOTE_USER}@${REMOTE_HOST}" \
            "tail -f ${REMOTE_DIR}/logs/*_${JOB_ID}.out 2>/dev/null || \
             tail -f ${REMOTE_DIR}/logs/*_${JOB_ID}.err 2>/dev/null || \
             echo 'Log not found yet — job may still be queued.'"
        ;;
    *)
        echo "Usage: $0 [--history | <JOBID>]"
        exit 1
        ;;
esac
