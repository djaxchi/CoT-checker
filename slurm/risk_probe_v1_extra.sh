#!/bin/bash
# risk_probe_v1 EXTRA_STEPS, run by bidirectional_token_probe_train_tamia.sh after the
# xfit5 fits, inside the same allocation and venv: rescore risk_probe_v0's three
# xfit_v2ds checkpoints through this job's store (reproduction gate, E1 validity
# score) and dump the before/after/mean rows of every label step and fork view.
set -euo pipefail
: "${STORE_DIR:?}" "${OUT_ROOT:?}" "${MANIFEST:?}"
V0=/project/aip-azouaq/$USER/cot_mech/risk_probe_v0/prod/fits
ck=()
for s in 42 43 44; do ck+=("xfit_v2ds_s${s}=$V0/xfit_v2ds_s${s}/local/best.pt"); done
CUDA_VISIBLE_DEVICES=0 python scripts/factorized_step_labels/score_with_checkpoints.py --store "$STORE_DIR" \
  --ckpt "${ck[@]}" --splits fsl_forks gate_rp_test --out_dir "$OUT_ROOT/existing_scores"
python scripts/risk_probe_v1/dump_rows.py --store "$STORE_DIR" --lab_views "$MANIFEST/lab_views.jsonl" \
  --out "$OUT_ROOT/features/rows_L35.npz"
