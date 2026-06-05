#!/bin/bash
# ---------------------------------------------------------------------------
# Sprint 2: fork dataset statistics / manifest inspection.
#
# Pretty-prints the fork build manifest: valid forks, split sizes, sibling
# distribution, and pair counts under BOTH one/all modes (so the one-vs-all
# choice stays grounded in real numbers). CPU/IO only, no GPU. Run directly on
# the login node; do NOT submit to the GPU queue.
#
# Run:  bash scripts/tamia/jobs/s2/stats_forks.sh [FORKS_DIR]
# ---------------------------------------------------------------------------

set -euo pipefail

SCRATCH_BASE="${SCRATCH:-/scratch/$USER}/cot_mech"
FORKS_DIR="${1:-$SCRATCH_BASE/s2_forks/data}"
MANIFEST="$FORKS_DIR/forks_manifest.json"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    echo "Run: bash scripts/tamia/jobs/s2/build_forks.sh first."
    exit 1
fi

python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print("="*55)
print("  Sprint 2 fork dataset statistics")
print("="*55)
print(f"  run_name              : {m['run_name']}")
print(f"  pair_mode             : {m['pair_mode']}")
print(f"  valid forks (total)   : {m['n_valid_forks_total']}")
print(f"  problem-id overlap    : {m['problem_id_overlap_train_val']}  (must be 0)")
print("  --- split counts ---")
for k, v in m["counts"].items():
    print(f"  {k:34s}: {v}")
print("  --- sibling distribution ---")
for k, v in m["sibling_distribution"].items():
    print(f"  {k:34s}: {v}")
print("="*55)
PY
