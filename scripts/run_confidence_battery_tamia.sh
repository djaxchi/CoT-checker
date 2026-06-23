#!/bin/bash
# Stage 0 A/B battery on every size, then a cross-size table of how much of the
# probe's AUC lift the model-confidence features remove. Needs the fork hidden
# encodes, the confidence sidecar (run_fork_confidence_sweep.sh), and per-size
# linear_probe.pt. CPU-only and light.
#
# Run from repo root on Tamia:  bash scripts/run_confidence_battery_tamia.sh
set -uo pipefail

REPO="${REPO:-$PWD}"; cd "$REPO" || { echo "run from repo root"; exit 1; }
RUNS_ROOT="${RUNS_ROOT:-runs/s1_model_size_dense}"
OUT_ROOT="${OUT_ROOT:-runs/fork_rep_audit}"
STEM="${STEM:-forks_val_items}"
ORDER="${ORDER:-qwen2_5_1_5b qwen2_5_3b qwen2_5_7b qwen2_5_14b qwen2_5_32b}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "[env] modules + venv ..."
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0 2>/dev/null
ENVDIR="${SLURM_TMPDIR:-/tmp}/confidence_battery_env"
virtualenv --no-download "$ENVDIR" >/dev/null 2>&1 || python -m venv "$ENVDIR"
# shellcheck disable=SC1091
source "$ENVDIR/bin/activate"
pip install --no-index --upgrade pip >/dev/null 2>&1
pip install --no-index numpy torch >/dev/null 2>&1 \
  || { echo "[env] FATAL: need numpy/torch --no-index"; exit 1; }
PLOTS_FLAG=""
pip install --no-index matplotlib >/dev/null 2>&1 || PLOTS_FLAG="--no_plots"

printf '\n%-16s %-9s %-12s %-9s %s\n' TAG "AUC_raw" "AUC_no_conf" "frac_rm" verdict
echo "------------------------------------------------------------------------------"
ROWS="$OUT_ROOT/_confidence_battery_cross_size.csv"
mkdir -p "$OUT_ROOT"
echo "tag,auc_raw,auc_no_confidence,frac_lift_removed,verdict" > "$ROWS"

for TAG in $ORDER; do
  H="$RUNS_ROOT/$TAG/forks/${STEM}_h.npy"
  CONF="$RUNS_ROOT/$TAG/forks/${STEM}_confidence.jsonl"
  PROBE="$RUNS_ROOT/$TAG/linear_probe.pt"
  OUT="$OUT_ROOT/$TAG/confidence_battery"
  if [[ ! -f "$H" || ! -f "$CONF" || ! -f "$PROBE" ]]; then
    printf '%-16s %s\n' "$TAG" "MISSING h/confidence/probe (run fork encode + confidence sweep)"
    continue
  fi
  python scripts/analyze_confidence_battery.py --h "$H" --confidence "$CONF" \
    --probe "$PROBE" --out "$OUT" $PLOTS_FLAG >/dev/null 2>&1 \
    || { printf '%-16s %s\n' "$TAG" "ERROR"; continue; }
  read -r AR ANC FR VB < <(python - "$OUT/metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
fr = m["frac_probe_lift_removed_by_confidence"]
vb = "NOT-surprise" if fr < 0.25 else "partial" if fr < 0.60 else "IS-surprise"
print(f'{m["auc_probe_raw"]:.3f} {m["auc_probe_after_removing_all_confidence"]:.3f} '
      f'{fr:.2f} {vb}')
PY
)
  printf '%-16s %-9s %-12s %-9s %s\n' "$TAG" "$AR" "$ANC" "$FR" "$VB"
  echo "$TAG,$AR,$ANC,$FR,$VB" >> "$ROWS"
done
echo "------------------------------------------------------------------------------"
echo "[ok] cross-size table -> $ROWS"
echo "[ok] per-size report  -> $OUT_ROOT/<tag>/confidence_battery/confidence_battery_report.md"
echo "[ok] GATE A: frac_rm < 0.25 at every size -> proceed to Stage 1 (on-policy)."
echo "[ok]         frac_rm > 0.60 -> the probe is largely a surprise meter."
