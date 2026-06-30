#!/bin/bash
#SBATCH --job-name=s1ms_ml_wdsweep
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
# NOTE: TamIA allocates h100 GPUs by whole node, so we must request h100:4 even though
# this sweep only re-trains a linear probe on cached features (no model load, no encode).
#
# Weight-decay sweep for the all-layer (concat, 100k-dim) DenseLinear probe.
#
# The all-layer run (run_multilayer_dense_7b.sh) trained with weight_decay=0.0 and got
# oracle macro F1 0.433 / val 0.178 (vs last-layer 0.413 / 0.237). The wide concat needs
# L2 (the offline bake-off needed C=0.05). This re-trains the SAME probe at several L2
# strengths and re-evaluates PB-F1, REUSING the already-encoded+assembled caches:
#     merged/{probe_train_40k,val_1k}_h.npy   and   processbench_eval_shards/<subset>/pb_step_h.npy
# So every sweep point is a seconds-long re-train + re-eval, no GPU encode, no re-assemble.
#
# Submit:  sbatch slurm/s1_model_size_multilayer/sweep_weight_decay_7b.sh
set -euo pipefail

TAG="${TAG:-qwen2_5_7b}"
REPO_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/CoT-checker}}"
S1MS_DIR="${S1MS_DIR:-$REPO_ROOT/slurm/s1_model_size}"
[[ -f "$S1MS_DIR/models.env" ]] || { echo "[wd] FATAL: cannot find $S1MS_DIR/models.env" >&2; exit 1; }
# shellcheck disable=SC1091
source "$S1MS_DIR/models.env"
# shellcheck disable=SC1091
source "$S1MS_DIR/_common.sh"

MODEL_ID="${S1MS_MODEL_ID[$TAG]}"
MODEL_DIR="$RUNS_ROOT/${TAG}_multilayer"
MERGED="$MODEL_DIR/merged"
PB_EVAL="$MODEL_DIR/processbench_eval_shards"
SWEEP_DIR="$MODEL_DIR/wd_sweep"
LOG_DIR="$MODEL_DIR/logs"
mkdir -p "$SWEEP_DIR" "$LOG_DIR"
LOG="$LOG_DIR/wd_sweep.log"

WDS="${WDS:-0.0 1e-4 1e-3 1e-2 1e-1 1.0}"
BASE_ORACLE="${BASE_ORACLE:-0.4132}"
BASE_VAL="${BASE_VAL:-0.2369}"

# Preconditions: the assembled caches from the main run must exist.
[[ -f "$MERGED/probe_train_40k_h.npy" && -f "$MERGED/val_1k_h.npy" ]] || {
  echo "[wd] FATAL: missing assembled PRM cache in $MERGED (run run_multilayer_dense_7b.sh first)" >&2; exit 1; }
for s in "${S1MS_SUBSETS[@]}"; do
  [[ -f "$PB_EVAL/$s/pb_step_h.npy" ]] || {
    echo "[wd] FATAL: missing assembled PB cache $PB_EVAL/$s/pb_step_h.npy" >&2; exit 1; }
done

s1ms_env_setup
echo "[wd] TAG=$TAG sweep WDS=[$WDS]  reuse caches in $MODEL_DIR" | tee "$LOG"
s1ms_venv

for wd in $WDS; do
  WDDIR="$SWEEP_DIR/wd_${wd}"
  mkdir -p "$WDDIR/eval"
  echo "[wd] === weight_decay=$wd -> $WDDIR ===" | tee -a "$LOG"

  # ---- train the SAME DenseLinear probe on the concat, only L2 changes -------
  python scripts/s1ms_train_dense_probe.py \
    --cache_dir "$MERGED" --out_dir "$WDDIR" \
    --probe_train_stem probe_train_40k --val_stem val_1k \
    --model_name "$MODEL_ID" --seed 42 --weight_decay "$wd" 2>&1 | tee -a "$LOG"

  # ---- re-evaluate PB-F1 per subset on the existing assembled features -------
  for subset in "${S1MS_SUBSETS[@]}"; do
    out="$WDDIR/eval/$subset"; mkdir -p "$out"
    python scripts/evaluate_saved_probe_on_processbench.py \
      --probe "$WDDIR/linear_probe.pt" \
      --pb_latents "$PB_EVAL/$subset/pb_step_h.npy" --pb_meta "$PB_EVAL/$subset/pb_step_meta.jsonl" \
      --threshold_json "$WDDIR/threshold.json" \
      --also_oracle --oracle_threshold_step 0.005 \
      --method_name dense_linear --pb_subset "$subset" \
      --out_json "$out/metrics.json" 2>&1 | tee -a "$LOG"
  done

  # ---- aggregate to per_subset_metrics.json (SAME aggregator) ---------------
  python scripts/s1ms_aggregate_model.py \
    --eval_dir "$WDDIR/eval" --out_dir "$WDDIR" \
    --subsets "${S1MS_SUBSETS[@]}" 2>&1 | tee -a "$LOG"
done

# ---- final comparison table --------------------------------------------------
echo "================ weight-decay sweep: macro F1 (oracle / val) ============" | tee -a "$LOG"
python - "$SWEEP_DIR" "$BASE_ORACLE" "$BASE_VAL" $WDS <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
sweep = Path(sys.argv[1]); base_o = float(sys.argv[2]); base_v = float(sys.argv[3])
wds = sys.argv[4:]
rows = []
for wd in wds:
    f = sweep / f"wd_{wd}" / "per_subset_metrics.json"
    if not f.exists():
        print(f"  wd={wd:>6}  (missing)"); continue
    m = json.load(open(f))
    rows.append((wd, m["macro_f1_oracle"], m["macro_f1_val_threshold"]))
print(f"  {'weight_decay':>12} | {'oracle':>7} {'d_oracle':>9} | {'val':>7} {'d_val':>8}")
print(f"  {'last-layer':>12} | {base_o:7.4f} {'  base':>9} | {base_v:7.4f} {'  base':>8}")
for wd, o, v in rows:
    print(f"  {wd:>12} | {o:7.4f} {o-base_o:+9.4f} | {v:7.4f} {v-base_v:+8.4f}")
if rows:
    bo = max(rows, key=lambda r: r[1]); bv = max(rows, key=lambda r: r[2])
    print(f"\n  best oracle: wd={bo[0]}  oracle={bo[1]:.4f} ({bo[1]-base_o:+.4f} vs last-layer)")
    print(f"  best val   : wd={bv[0]}  val={bv[2]:.4f} ({bv[2]-base_v:+.4f} vs last-layer)")
PY
echo "[wd] done -> $SWEEP_DIR/*/per_subset_metrics.json" | tee -a "$LOG"
