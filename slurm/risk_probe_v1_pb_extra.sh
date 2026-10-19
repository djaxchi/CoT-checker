#!/bin/bash
# risk_probe_v1 on ProcessBench (post hoc), EXTRA_STEPS of bidirectional_token_probe_train_tamia.sh
# after encoding the ProcessBench `full` step views ($R/pb_views): score them with the
# five xfit5 fold models and §21.18's full_v2ds fits, dump their L35 rows, then fit the
# readouts on all label steps and score ProcessBench (scripts/risk_probe_v1/score_pb.py).
set -euo pipefail
: "${STORE_DIR:?}" "${OUT_ROOT:?}"
C=/project/aip-azouaq/$USER/cot_mech
R=$C/risk_probe_v1
SVL=$C/step_value_labels_v1
CAV3=$C/context_ablation_v3
SPLITS=(pb_gsm8k pb_math pb_olympiadbench pb_omnimath)
ck=()
for k in 0 1 2 3 4; do ck+=("xfit5_f$k=$R/prod/fits/xfit5_f$k/local/best.pt"); done
for s in 42 43 44; do ck+=("full_v2ds_s$s=$CAV3/prod/fits/full_v2ds_s$s/local/best.pt"); done
CUDA_VISIBLE_DEVICES=0 python scripts/factorized_step_labels/score_with_checkpoints.py --store "$STORE_DIR" \
  --ckpt "${ck[@]}" --splits "${SPLITS[@]}" --out_dir "$OUT_ROOT/pb_scores"
python scripts/risk_probe_v1/dump_rows.py --store "$STORE_DIR" --splits "${SPLITS[@]}" \
  --out "$OUT_ROOT/features/pb_rows_L35.npz"
CUDA_VISIBLE_DEVICES=0 python scripts/risk_probe_v1/score_pb.py --label_rows "$R/prod/features/rows_L35.npz" \
  --lab_views "$R/views/lab_views.jsonl" --step_values "$SVL/label/summary/step_values.jsonl" \
  --trace_set "$SVL/split/trace_set_label.json" --readouts "$R/prod/readouts" \
  --e2_scores "$R/analysis_prod/step_scores.jsonl" --pb_rows "$OUT_ROOT/features/pb_rows_L35.npz" \
  --pb_meta "$CAV3/views/full/meta" --scores_dir "$OUT_ROOT/pb_scores" --saved_v2ds "$CAV3/prod/fits" \
  --out_dir "$OUT_ROOT/analysis"
