#!/usr/bin/env bash
# Sync code to TamIA then submit a Slurm script.
# Usage: submit.sh <slurm_script> [extra sbatch args...]
#
# Example:
#   submit.sh scripts/slurm/tamia_train_probe.sh
#   submit.sh scripts/slurm/tamia_generate_data.sh --time=06:00:00

set -euo pipefail

REMOTE_USER="${TAMIA_USER:-$USER}"
REMOTE_HOST="tamia.alliancecan.ca"
REMOTE_DIR="~/CoT-checker"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <slurm_script> [extra sbatch args...]"
    exit 1
fi

SLURM_SCRIPT="$1"
shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[submit] syncing code..."
bash "${SCRIPT_DIR}/sync.sh"

echo "[submit] creating logs/ dir on cluster..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}/logs"

REMOTE_SCRIPT="${REMOTE_DIR}/${SLURM_SCRIPT}"
echo "[submit] submitting ${SLURM_SCRIPT}..."
JOB_OUTPUT=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" \
    "cd ${REMOTE_DIR} && sbatch ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"} ${REMOTE_SCRIPT}")

echo "[submit] $JOB_OUTPUT"

JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+$')
if [[ -n "$JOB_ID" ]]; then
    echo "[submit] job ID: $JOB_ID"
    echo "[submit] monitor with: bash scripts/tamia/monitor.sh $JOB_ID"
fi
