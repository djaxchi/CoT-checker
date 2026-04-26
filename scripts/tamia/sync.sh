#!/usr/bin/env bash
# Sync local code to TamIA login node.
# Usage: sync.sh [--dry-run]
#
# Flow: laptop → cluster (code only).
# Never rsync results, checkpoints, or data from laptop to cluster.

set -euo pipefail

REMOTE_USER="${TAMIA_USER:-$USER}"
REMOTE_HOST="tamia.alliancecan.ca"
REMOTE_DIR="~/CoT-checker"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

RSYNC_OPTS=(-avz --delete
    --exclude='.git/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='results/'
    --exclude='data/'
    --exclude='*.pt'
    --exclude='*.npz'
    --exclude='*.safetensors'
    --exclude='notebooks/.ipynb_checkpoints/'
    --exclude='cot_checker.egg-info/'
    --exclude='.skills/'
    --exclude='memory/'
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    RSYNC_OPTS+=(--dry-run)
    echo "[sync] DRY RUN — no files will be transferred"
fi

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

echo "[sync] ${REPO_ROOT}/ → ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
rsync "${RSYNC_OPTS[@]}" \
    --include='src/***' \
    --include='scripts/***' \
    --include='experiments/***' \
    --include='tests/***' \
    --include='pyproject.toml' \
    --include='uv.lock' \
    --exclude='*' \
    "${REPO_ROOT}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "[sync] done"
