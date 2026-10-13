#!/bin/bash
# risk_probe_v0 EXTRA_STEPS, run by bidirectional_token_probe_train_tamia.sh after the
# xfit fits, inside the same allocation and venv: rescore the six §21.18 checkpoints
# through this job's store (reproduction gate + Q1) and dump pooled fork states (Q2).
set -euo pipefail
: "${STORE_DIR:?}" "${OUT_ROOT:?}"
CAV3=/project/aip-azouaq/$USER/cot_mech/context_ablation_v3/prod/fits
ck=()
for src in v2ds v1human; do for s in 42 43 44; do
  ck+=("full_${src}_s${s}=$CAV3/full_${src}_s${s}/local/best.pt")
done; done
CUDA_VISIBLE_DEVICES=0 python scripts/factorized_step_labels/score_with_checkpoints.py --store "$STORE_DIR" \
  --ckpt "${ck[@]}" --splits fsl_forks gate_rp_test --out_dir "$OUT_ROOT/existing_scores"
python scripts/factorized_step_labels/dump_fork_features.py --store "$STORE_DIR" --split fsl_forks \
  --out "$OUT_ROOT/features/fsl_forks_L35.npz"
