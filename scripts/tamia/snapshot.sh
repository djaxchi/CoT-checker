#!/usr/bin/env bash
# Push an isolated, immutable source snapshot to TamIA (code only).
# Usage: snapshot.sh <experiment_id> [--dry-run]
# Prints the remote snapshot path on the last line.
#
# Unlike sync.sh this never touches the shared ~/CoT-checker checkout (no
# --delete against a tree other jobs run from), includes slurm/ and docs/, and
# records provenance without committing: git HEAD, a hash of `git diff HEAD`,
# the list of untracked files shipped, and a sha256 of the shipped tree.
set -euo pipefail
EXP="${1:?experiment id}"
DRY="${2:-}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
HEAD="$(git -C "$REPO" rev-parse HEAD)"
DIFF_SHA="$(git -C "$REPO" diff HEAD | shasum -a 256 | cut -c1-16)"
TAG="$(date +%Y%m%d-%H%M%S)-${HEAD:0:7}-${DIFF_SHA:0:8}"
# SNAP_HOST=rorqual ships to Rorqual scratch instead of TamIA project storage
SNAP_HOST="${SNAP_HOST:-tamia}"
if [[ "$SNAP_HOST" == tamia ]]; then
  REMOTE_BASE="/project/aip-azouaq/dchikhi/cot_mech/$EXP/snapshots"
else
  REMOTE_BASE="/scratch/dchikhi/cot_mech/$EXP/snapshots"
fi
DEST="$REMOTE_BASE/$TAG"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
rsync -a --exclude='__pycache__/' --exclude='*.pyc' \
  --include='src/***' --include='scripts/***' --include='slurm/***' --include='experiments/***' \
  --include='tests/***' --include='docs/' --include='docs/*.md' --include='pyproject.toml' \
  --exclude='*' "$REPO/" "$STAGE/"
TREE_SHA="$(cd "$STAGE" && find . -type f | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | cut -c1-16)"
{
  echo "experiment: $EXP"
  echo "snapshot: $TAG"
  echo "git_head: $HEAD"
  echo "git_diff_head_sha256_16: $DIFF_SHA"
  echo "tree_sha256_16: $TREE_SHA"
  echo "created: $(date -u +%FT%TZ)"
  echo "untracked_shipped:"
  git -C "$REPO" ls-files --others --exclude-standard -- src scripts slurm experiments tests docs | sed 's/^/  /'
} > "$STAGE/SNAPSHOT_INFO"
git -C "$REPO" diff HEAD > "$STAGE/SNAPSHOT_DIFF.patch"

if [[ "$DRY" == "--dry-run" ]]; then
  echo "[dry-run] would ship $(find "$STAGE" -type f | wc -l) files to $SNAP_HOST:$DEST"
  for p in slurm/bidirectional_token_probe_train_tamia.sh docs/bidirectional_token_probe_v1_plan.md \
           scripts/train_contextual_token_probe.py src/probes/contextual_token_probe.py; do
    [[ -f "$STAGE/$p" ]] && echo "  included: $p" || echo "  MISSING: $p"
  done
  exit 0
fi
ssh "$SNAP_HOST" "mkdir -p '$REMOTE_BASE'"
rsync -az "$STAGE/" "$SNAP_HOST:$DEST/"
ssh "$SNAP_HOST" "chmod -R a-w '$DEST'"
echo "$DEST"
