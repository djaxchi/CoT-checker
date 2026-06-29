#!/bin/bash
#SBATCH --job-name=s1ms_ml_dense
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#
# Repeat the S1 small-scale DenseLinear experiment (probe_train_40k / val_1k ->
# ProcessBench PB-F1) with ALL LAYERS COMBINED instead of just the last layer.
#
# Same data, same probe trainer, same threshold logic, same PB-F1 evaluator, same
# aggregator as the original (runs/s1_model_size_dense/qwen2_5_7b, oracle macro
# F1=0.413 / val=0.237). The ONLY changes:
#   * encode every layer (one forward pass, output_hidden_states) instead of last
#   * assemble_multilayer_concat.py concatenates + z-scores the layers into the
#     same {stem}_h.npy contract the downstream stages already consume
#
# One self-contained job: small scale (40k train) does not need the staged DAG.
# Reuses the frozen PRM800K split and the existing python stages verbatim.
#
# Submit:  sbatch slurm/s1_model_size_multilayer/run_multilayer_dense_7b.sh
# Compare: runs/s1_model_size_dense/qwen2_5_7b_multilayer/per_subset_metrics.json
#          vs runs/s1_model_size_dense/qwen2_5_7b/per_subset_metrics.json
set -euo pipefail

TAG="${TAG:-qwen2_5_7b}"
# sbatch copies this script to a spool dir, so ${BASH_SOURCE} does NOT point at the
# repo. Resolve the checkout from the submit dir (where `sbatch` was invoked) with a
# $HOME fallback, then find the shared S1 config relative to it.
REPO_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/CoT-checker}}"
S1MS_DIR="${S1MS_DIR:-$REPO_ROOT/slurm/s1_model_size}"
[[ -f "$S1MS_DIR/models.env" ]] || { echo "[ml] FATAL: cannot find $S1MS_DIR/models.env; set PROJECT_ROOT or S1MS_DIR" >&2; exit 1; }
# shellcheck disable=SC1091
source "$S1MS_DIR/models.env"
# shellcheck disable=SC1091
source "$S1MS_DIR/_common.sh"

MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
BS="${S1MS_BATCH[$TAG]}"
MODEL_DIR="$RUNS_ROOT/${TAG}_multilayer"
PRM_ML="$MODEL_DIR/prm_multilayer"
PB_ML="$MODEL_DIR/pb_multilayer"
MERGED="$MODEL_DIR/merged"
PB_EVAL="$MODEL_DIR/processbench_eval_shards"
LOG_DIR="$MODEL_DIR/logs"
SCALER="$MERGED/scaler.npz"
export S1MS_MODEL_ID_CUR="$MODEL_ID" S1MS_STAGE="multilayer_dense"
mkdir -p "$PRM_ML" "$PB_ML" "$MERGED" "$PB_EVAL" "$LOG_DIR"
LOG="$LOG_DIR/run_multilayer_dense.log"

# All 28 transformer layers (hidden_states indices 1..N): fracs i/N round to i.
# Generated with awk (always on PATH) so this does not depend on `module load python`,
# which has not run yet at this point in the script.
NLAYERS="${NLAYERS:-28}"
LAYER_FRACS="$(awk -v n="$NLAYERS" 'BEGIN{for(i=1;i<=n;i++) printf "%.6f ", i/n}')"

s1ms_env_setup
echo "[ml] TAG=$TAG model=$MODEL_ID layers=all($NLAYERS) bs=$BS" | tee "$LOG"
echo "[ml] layer_fracs=$LAYER_FRACS" | tee -a "$LOG"
s1ms_ensure_model_cached "$MODEL_ID" | tee -a "$LOG"
s1ms_venv
pip install --no-index scikit-learn >/dev/null 2>&1 || true   # not needed; assemble is numpy-only

for f in "$PRM_TRAIN_JSONL" "$PRM_VAL_JSONL"; do
  [[ -f "$PRM_SPLIT_DIR/$f" ]] || { echo "[ml] FATAL: missing $PRM_SPLIT_DIR/$f" | tee -a "$LOG" >&2; exit 1; }
done

# ---- 1. Multilayer-encode PRM800K train+val (one GPU; 40k is small) ----------
echo "[ml] (1) encode PRM800K all layers -> $PRM_ML" | tee -a "$LOG"
(
  export CUDA_VISIBLE_DEVICES=0
  run_with_oom_retry "$BS" "$LOG_DIR/enc_prm.log" \
    python scripts/encode_prm800k_multilayer.py \
      --data_dir "$PRM_SPLIT_DIR" --out_dir "$PRM_ML" \
      --model_name_or_path "$MODEL_ID" --local_files_only \
      --run_name "s1ms_ml_${TAG}_prm" \
      --splits "${PRM_TRAIN_JSONL}:probe_train_40k" "${PRM_VAL_JSONL}:val_1k" \
      --layer_fracs $LAYER_FRACS \
      --max_seq_len -1 --batch_size __BS__ --model_dtype float16 --save_dtype float16 --force
) || { echo "[ml] FATAL: PRM encode failed" | tee -a "$LOG" >&2; exit 1; }

# ---- 2. Multilayer-encode the 4 ProcessBench subsets (one GPU each) ----------
echo "[ml] (2) encode ProcessBench all layers -> $PB_ML" | tee -a "$LOG"
pids=(); g=0
for subset in "${S1MS_SUBSETS[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES="$g"
    raw="$(s1ms_resolve_pb_raw "$subset")"
    run_with_oom_retry "$BS" "$LOG_DIR/enc_pb_${subset}.log" \
      python scripts/encode_processbench_multilayer.py \
        --raw_file "$raw" --out_dir "$PB_ML/$subset" \
        --model_name_or_path "$MODEL_ID" --local_files_only \
        --subset_name "$subset" --layer_fracs $LAYER_FRACS \
        --max_seq_len -1 --batch_size __BS__ --model_dtype float16 --save_dtype float16 --force
  ) & pids+=("$!"); g=$((g + 1))
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || fail=1; done
[[ $fail -eq 0 ]] || { echo "[ml] FATAL: a PB encode failed" | tee -a "$LOG" >&2; exit 1; }

# ---- 3. Assemble: fit z-score on train, apply to val + every PB subset --------
echo "[ml] (3) assemble concat + standardize" | tee -a "$LOG"
python scripts/assemble_multilayer_concat.py --mode fit \
  --in_dir "$PRM_ML" --stem probe_train_40k --out_dir "$MERGED" --scaler "$SCALER" \
  --save_dtype float16 2>&1 | tee -a "$LOG"
python scripts/assemble_multilayer_concat.py --mode apply \
  --in_dir "$PRM_ML" --stem val_1k --out_dir "$MERGED" --scaler "$SCALER" \
  --save_dtype float16 2>&1 | tee -a "$LOG"
for subset in "${S1MS_SUBSETS[@]}"; do
  python scripts/assemble_multilayer_concat.py --mode apply \
    --in_dir "$PB_ML/$subset" --stem pb_step --out_dir "$PB_EVAL/$subset" --scaler "$SCALER" \
    --save_dtype float16 2>&1 | tee -a "$LOG"
done

# ---- 4. Train the SAME DenseLinear probe on the concatenated features ---------
echo "[ml] (4) train dense_linear on concat" | tee -a "$LOG"
python scripts/s1ms_dump_model_config.py \
  --model_name_or_path "$MODEL_ID" --local_files_only \
  --params_label "${S1MS_PARAMS_LABEL[$TAG]}" \
  --out_json "$MODEL_DIR/model_config.json" 2>&1 | tee -a "$LOG"
python scripts/s1ms_train_dense_probe.py \
  --cache_dir "$MERGED" --out_dir "$MODEL_DIR" \
  --probe_train_stem probe_train_40k --val_stem val_1k \
  --model_name "$MODEL_ID" --seed 42 2>&1 | tee -a "$LOG"

# ---- 5. Evaluate PB-F1 per subset (SAME evaluator) ---------------------------
echo "[ml] (5) evaluate PB-F1 per subset" | tee -a "$LOG"
for subset in "${S1MS_SUBSETS[@]}"; do
  out="$PB_EVAL/$subset"
  python scripts/evaluate_saved_probe_on_processbench.py \
    --probe "$MODEL_DIR/linear_probe.pt" \
    --pb_latents "$out/pb_step_h.npy" --pb_meta "$out/pb_step_meta.jsonl" \
    --threshold_json "$MODEL_DIR/threshold.json" \
    --also_oracle --oracle_threshold_step 0.005 \
    --method_name dense_linear --pb_subset "$subset" \
    --out_json "$out/metrics.json" \
    --out_scores_jsonl "$out/step_scores.jsonl" \
    --out_predictions_jsonl "$out/predictions.jsonl" 2>&1 | tee -a "$LOG"
done

# ---- 6. Aggregate to per_subset_metrics.json (SAME aggregator) ---------------
echo "[ml] (6) aggregate" | tee -a "$LOG"
python scripts/s1ms_aggregate_model.py \
  --eval_dir "$PB_EVAL" --out_dir "$MODEL_DIR" \
  --subsets "${S1MS_SUBSETS[@]}" 2>&1 | tee -a "$LOG"

echo "================ all-layer vs last-layer (oracle / val macro F1) ============" | tee -a "$LOG"
python - "$MODEL_DIR/per_subset_metrics.json" "$RUNS_ROOT/$TAG/per_subset_metrics.json" <<'PY' | tee -a "$LOG"
import json, sys
ml = json.load(open(sys.argv[1]))
try:
    base = json.load(open(sys.argv[2]))
except FileNotFoundError:
    base = {"macro_f1_oracle": 0.4132, "macro_f1_val_threshold": 0.2369}
print(f"  all-layer : oracle={ml['macro_f1_oracle']:.4f}  val={ml['macro_f1_val_threshold']:.4f}")
print(f"  last-layer: oracle={base['macro_f1_oracle']:.4f}  val={base['macro_f1_val_threshold']:.4f}")
print(f"  delta     : oracle={ml['macro_f1_oracle']-base['macro_f1_oracle']:+.4f}  "
      f"val={ml['macro_f1_val_threshold']-base['macro_f1_val_threshold']:+.4f}")
PY
echo "[ml] done -> $MODEL_DIR/per_subset_metrics.json" | tee -a "$LOG"
