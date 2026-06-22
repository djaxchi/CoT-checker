#!/bin/bash
# Drive the L28/last fork representation audit on Tamia, 7B first then all sizes.
# CPU-only and light (~1k forks); runs inline on a login node or inside a job.
#
# Run from the repo root on Tamia:
#   bash scripts/run_fork_audit_tamia.sh
#
# Per size it needs (deterministic paths from the s1_model_size_dense pipeline):
#   probe : runs/s1_model_size_dense/<tag>/linear_probe.pt        (L28/last DenseLinear)
#   forks : runs/s1_model_size_dense/<tag>/forks/forks_val_items_{h.npy,meta.jsonl}
#   items : $FORK_ITEMS (default below; built by build_prm800k_forks.py)
# Sizes lacking a forks/ encode are skipped with a clear message (run the encode
# sweep first: slurm/s1_model_size/run_fork_encode_sweep.sh).
set -uo pipefail

REPO="${REPO:-$PWD}"
cd "$REPO" || { echo "run from repo root"; exit 1; }
FORK_ITEMS="${FORK_ITEMS:-/scratch/d/dchikhi/cot_mech/s2_forks/data/forks_val_items.jsonl}"
RUNS_ROOT="${RUNS_ROOT:-runs/s1_model_size_dense}"
OUT_ROOT="${OUT_ROOT:-runs/fork_rep_audit}"
ORDER="${ORDER:-qwen2_5_7b qwen2_5_1_5b qwen2_5_3b qwen2_5_14b qwen2_5_32b}"
NULL_ITER="${NULL_ITER:-300}"

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# ---- environment: build a light venv; plots are optional (sklearn+matplotlib).
echo "[env] loading modules + venv ..."
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0 2>/dev/null
ENVDIR="${SLURM_TMPDIR:-/tmp}/fork_audit_env"
virtualenv --no-download "$ENVDIR" >/dev/null 2>&1 || python -m venv "$ENVDIR"
# shellcheck disable=SC1091
source "$ENVDIR/bin/activate"
pip install --no-index --upgrade pip >/dev/null 2>&1
pip install --no-index numpy torch >/dev/null 2>&1 || { echo "[env] FATAL: numpy/torch unavailable --no-index"; exit 1; }
PLOTS_FLAG=""
if pip install --no-index scikit-learn matplotlib >/dev/null 2>&1; then
  echo "[env] sklearn+matplotlib available -> full plots"
else
  echo "[env] sklearn/matplotlib NOT available --no-index -> --no_plots (metrics + tables only)"
  PLOTS_FLAG="--no_plots"
fi

if [[ ! -f "$FORK_ITEMS" ]]; then
  echo "[audit] FATAL: fork items not found: $FORK_ITEMS"
  echo "        Set FORK_ITEMS=... (S2 forks_val_items.jsonl from build_prm800k_forks.py)."
  exit 1
fi

mkdir -p "$OUT_ROOT"
printf '\n%-16s %-8s %-8s %-10s %-8s %s\n' TAG FORKS PROBE "P(n>p)" "tru/fake" VERDICT
echo "--------------------------------------------------------------------------"

ran=0
for TAG in $ORDER; do
  MDIR="$RUNS_ROOT/$TAG"
  H="$MDIR/forks/forks_val_items_h.npy"
  META="$MDIR/forks/forks_val_items_meta.jsonl"
  PROBE="$MDIR/linear_probe.pt"
  OUT="$OUT_ROOT/$TAG"

  if [[ ! -f "$H" || ! -f "$META" ]]; then
    printf '%-16s %-8s %-8s %-10s %-8s %s\n' "$TAG" "MISSING" "-" "-" "-" "skip (no fork encode)"
    continue
  fi
  PROBE_ARG=()
  PSTATE="no"
  [[ -f "$PROBE" ]] && { PROBE_ARG=(--probe "$PROBE"); PSTATE="yes"; }

  echo "[audit] >>> $TAG  (probe=$PSTATE)"  >&2
  python scripts/analyze_fork_representation_audit.py \
    --fork_items "$FORK_ITEMS" --h "$H" --meta "$META" "${PROBE_ARG[@]}" \
    --out "$OUT" --null_iter "$NULL_ITER" $PLOTS_FLAG  >&2 || {
      printf '%-16s %-8s %-8s %-10s %-8s %s\n' "$TAG" "ok" "$PSTATE" "-" "-" "ERROR (see log above)"; continue; }

  # pull headline numbers out of metrics.json for the status table
  read -r PNP TRF VERD < <(python - "$OUT/metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
pnp = m.get("probe", {}).get("P_neg_gt_pos", float("nan"))
trf = m.get("controls", {}).get("true_over_fake_energy", float("nan"))
# recompute the green-light count the same way summary.md does
c = m.get("common", {}); cc = m.get("controls", {}); p = m.get("probe", {})
rows = []
if p:
    rows.append(p["P_neg_gt_pos"] > 0.65)
    rows.append(c["cos_muD_w"] > 0.15)
    rows.append(c["w_var_frac_delta"]/max(c["w_var_frac_raw_pooled"],1e-9) > 5)
rows.append(cc["true_over_fake_energy"] > 2)
rows.append(cc["signflip_mu_z"] > 3)
scm = m.get("surface_confounds_max", {"abs_r":0}); rows.append(scm["abs_r"] < 0.5)
npass = sum(bool(x) for x in rows)
verd = "GREEN" if npass >= max(3, len(rows)-1) else "HOLD"
print(f"{pnp:.3f} {trf:.2f} {verd}:{npass}/{len(rows)}")
PY
)
  printf '%-16s %-8s %-8s %-10s %-8s %s\n' "$TAG" "ok" "$PSTATE" "$PNP" "$TRF" "$VERD"
  ran=$((ran+1))
done

echo "--------------------------------------------------------------------------"
if [[ $ran -eq 0 ]]; then
  echo "[audit] no sizes had a fork encode. Run slurm/s1_model_size/run_fork_encode_sweep.sh first."
  exit 2
fi

# compact bundle for download (summaries + metrics + tables + any plots)
BUNDLE="$OUT_ROOT/fork_rep_audit_bundle.tar.gz"
tar -czf "$BUNDLE" -C "$OUT_ROOT" \
  $(cd "$OUT_ROOT" && find . -name 'summary.md' -o -name 'metrics.json' -o -path '*/tables/*' -o -path '*/plots/*') 2>/dev/null
echo "[audit] done: $ran size(s). Bundle -> $BUNDLE"
echo "[audit] inspect: cat $OUT_ROOT/qwen2_5_7b/summary.md"
