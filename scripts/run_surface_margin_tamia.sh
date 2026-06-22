#!/bin/bash
# Run the surface-explains-margin inspection on every size's pair_signatures.csv,
# then emit a cross-size surface-only R^2 table. Depends on the fork audit having
# already produced runs/fork_rep_audit/<tag>/tables/pair_signatures.csv.
#
# Run from repo root on Tamia:  bash scripts/run_surface_margin_tamia.sh
set -uo pipefail

REPO="${REPO:-$PWD}"; cd "$REPO" || { echo "run from repo root"; exit 1; }
AUDIT_ROOT="${AUDIT_ROOT:-runs/fork_rep_audit}"
ORDER="${ORDER:-qwen2_5_1_5b qwen2_5_3b qwen2_5_7b qwen2_5_14b qwen2_5_32b}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "[env] modules + venv ..."
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0 2>/dev/null
ENVDIR="${SLURM_TMPDIR:-/tmp}/surf_margin_env"
virtualenv --no-download "$ENVDIR" >/dev/null 2>&1 || python -m venv "$ENVDIR"
# shellcheck disable=SC1091
source "$ENVDIR/bin/activate"
pip install --no-index --upgrade pip >/dev/null 2>&1
pip install --no-index numpy pandas scipy scikit-learn matplotlib >/dev/null 2>&1 \
  || { echo "[env] FATAL: need numpy/pandas/scipy/scikit-learn/matplotlib --no-index"; exit 1; }
pip install --no-index tabulate >/dev/null 2>&1 || true   # nicer md tables; optional

printf '\n%-16s %-10s %-12s %s\n' TAG "surf_R2" "surf+geom" "interpretation"
echo "------------------------------------------------------------------------"
ROWS="runs/fork_rep_audit/_surface_margin_cross_size.csv"
echo "tag,surface_only_ridge_cv_r2,surface_only_std,surface_plus_geometry_cv_r2" > "$ROWS"

for TAG in $ORDER; do
  PS="$AUDIT_ROOT/$TAG/tables/pair_signatures.csv"
  OUT="$AUDIT_ROOT/$TAG/surface_margin"
  [[ -f "$PS" ]] || { printf '%-16s %s\n' "$TAG" "MISSING pair_signatures.csv"; continue; }
  python scripts/inspect_surface_margin.py --pair_signatures "$PS" --out "$OUT" >/dev/null 2>&1 || {
    printf '%-16s %s\n' "$TAG" "ERROR"; continue; }
  read -r R2 STD FULL BAND < <(python - "$OUT/surface_margin_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
so = s["surface_only"]; sg = s["surface_plus_geometry"]
r = so["ridge_cv_r2_mean"]
band = ("barely" if r < 0.05 else "weak" if r < 0.15 else "substantial" if r < 0.30 else "heavy")
print(f"{r:.4f} {so['ridge_cv_r2_std']:.4f} {sg['ridge_cv_r2_mean']:.4f} {band}")
PY
)
  printf '%-16s %-10s %-12s %s\n' "$TAG" "$R2" "$FULL" "$BAND"
  echo "$TAG,$R2,$STD,$FULL" >> "$ROWS"
done
echo "------------------------------------------------------------------------"
echo "[ok] cross-size table -> $ROWS"
echo "[ok] per-size report -> $AUDIT_ROOT/<tag>/surface_margin/surface_margin_report.md"
