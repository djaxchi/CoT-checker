#!/bin/bash
# Per-step margin-driver analysis on every size, then a cross-size table of how much
# of the score AUC is length. Needs the fork encodes + per-size linear_probe.pt.
#
# Run from repo root on Tamia:  bash scripts/run_margin_drivers_tamia.sh
set -uo pipefail

REPO="${REPO:-$PWD}"; cd "$REPO" || { echo "run from repo root"; exit 1; }
RUNS_ROOT="${RUNS_ROOT:-runs/s1_model_size_dense}"
OUT_ROOT="${OUT_ROOT:-runs/fork_rep_audit}"
FORK_ITEMS="${FORK_ITEMS:-/scratch/d/dchikhi/cot_mech/s2_forks/data/forks_val_items.jsonl}"
ORDER="${ORDER:-qwen2_5_1_5b qwen2_5_3b qwen2_5_7b qwen2_5_14b qwen2_5_32b}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "[env] modules + venv ..."
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0 2>/dev/null
ENVDIR="${SLURM_TMPDIR:-/tmp}/margin_drivers_env"
virtualenv --no-download "$ENVDIR" >/dev/null 2>&1 || python -m venv "$ENVDIR"
# shellcheck disable=SC1091
source "$ENVDIR/bin/activate"
pip install --no-index --upgrade pip >/dev/null 2>&1
pip install --no-index numpy torch scikit-learn matplotlib >/dev/null 2>&1 \
  || { echo "[env] FATAL: need numpy/torch/scikit-learn/matplotlib --no-index"; exit 1; }

[[ -f "$FORK_ITEMS" ]] || { echo "[drivers] FATAL: fork items not found: $FORK_ITEMS"; exit 1; }

printf '\n%-16s %-9s %-12s %-14s %s\n' TAG "AUC_raw" "AUC_no_len" "frac_len" verdict
echo "------------------------------------------------------------------------"
ROWS="$OUT_ROOT/_margin_drivers_cross_size.csv"
echo "tag,auc_raw,auc_no_length,auc_no_all,frac_lift_from_length" > "$ROWS"

for TAG in $ORDER; do
  H="$RUNS_ROOT/$TAG/forks/forks_val_items_h.npy"
  META="$RUNS_ROOT/$TAG/forks/forks_val_items_meta.jsonl"
  PROBE="$RUNS_ROOT/$TAG/linear_probe.pt"
  OUT="$OUT_ROOT/$TAG/margin_drivers"
  [[ -f "$H" && -f "$PROBE" ]] || { printf '%-16s %s\n' "$TAG" "MISSING encode/probe"; continue; }
  python scripts/inspect_margin_drivers.py --h "$H" --meta "$META" \
    --fork_items "$FORK_ITEMS" --probe "$PROBE" --out "$OUT" >/dev/null 2>&1 \
    || { printf '%-16s %s\n' "$TAG" "ERROR"; continue; }
  read -r AR ANL ANA FR VB < <(python - "$OUT/metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
fr = m["frac_auc_lift_from_length"]
vb = "NOT length" if fr < 0.25 else "partial" if fr < 0.6 else "LENGTH"
print(f'{m["auc_raw"]:.3f} {m["auc_after_removing_length"]:.3f} '
      f'{m["auc_after_removing_all_cheap_features"]:.3f} {fr:.2f} {vb}')
PY
)
  printf '%-16s %-9s %-12s %-14s %s\n' "$TAG" "$AR" "$ANL" "$FR" "$VB"
  echo "$TAG,$AR,$ANL,$ANA,$FR" >> "$ROWS"
done
echo "------------------------------------------------------------------------"
echo "[ok] cross-size table -> $ROWS"
echo "[ok] per-size report  -> $OUT_ROOT/<tag>/margin_drivers/margin_drivers_report.md"
echo "[ok] plots            -> $OUT_ROOT/<tag>/margin_drivers/plots/"
