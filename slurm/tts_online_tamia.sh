#!/bin/bash
#SBATCH --job-name=tts_online
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

# Job B of tts_roster_v1: step-level rejection, swept over threshold and scorer.
#
# §20.17 showed rejection pays (+0.098 over plain at 2.30x the tokens, blind
# control null at +0.024, p=0.49) with ONE scorer at two hand-picked quantiles,
# on a PRM800K pool. It never asked which scorer makes it work, and q=0.85's
# interval touched zero while q=0.70's did not, which says the operating point
# is at or below 0.70 and that nobody looked.
#
# Here the threshold is an axis and the scorer is an axis.
#
# CALIBRATION IS PER SCORER AND PER DATASET. A quantile of PRM800K step scores
# is a different rejection RATE on GSM8K, and a scorer that rejects 45% of steps
# where another rejects 30% has bought retries, not shown skill. Stage 0 runs the
# plain arm with --score_plain, which scores the step it keeps anyway and changes
# no decision, so each scorer's threshold comes from the distribution it will
# actually meet. That is also what makes this job independent of job A, so the
# two run side by side on eight GPUs instead of one after the other.
#
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/tts_roster_v1}"
SPLITS="${SPLITS:-$RUN_ROOT/splits}"
REPROBE_ROOT="${REPROBE_ROOT:-$SCRATCH/cot_mech/reprobe_v1}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/online}"
LAYER="${LAYER:-35}"
CALIB_PROBLEMS="${CALIB_PROBLEMS:-120}"
MAX_PROBLEMS="${MAX_PROBLEMS:-300}"
MAX_STEPS="${MAX_STEPS:-28}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_RETRIES="${MAX_RETRIES:-2}"
QUANTILES="${QUANTILES:-0.50 0.65 0.80}"
# Two probes that differ in representation, not just in seed, so "the verifier"
# is not one architecture's quirk.
CELLS="${CELLS:-$REPROBE_ROOT/cells/step_tokens__transformer_d256_l2_f1024_h4__seed42 $REPROBE_ROOT/cells/last_token__linear__seed42}"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
LOG_DIR="$RUN_ROOT/logs"; mkdir -p "$RUN_ROOT" "$LOG_DIR" "$OUT_DIR"
LOG="$LOG_DIR/tts_online-${SLURM_JOB_ID:-$$}.log"

[[ -f "$SPLITS/online_subset.jsonl" ]] || {
  echo "[FATAL] missing $SPLITS/online_subset.jsonl" >&2; exit 2; }
for C in $CELLS; do
  [[ -d "$C" ]] || { echo "[FATAL] no cell at $C" >&2; exit 2; }
done

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-tts_online}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
problems   : $SPLITS/online_subset.jsonl (first $MAX_PROBLEMS)
scorers    : $CELLS
quantiles  : $QUANTILES   retries: $MAX_RETRIES
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy

run_sharded () {   # $1 = tag, rest = args to online_bon.py
  local tag="$1"; shift
  local pids=()
  for i in $(seq 0 3); do
    CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/online_bon.py \
      --traces "$SPLITS/online_subset.jsonl" --id_field problem_id \
      --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
      --layer "$LAYER" --max_steps "$MAX_STEPS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --prompt_style fewshot --n_shot 4 \
      --plain_temperature 1.0 --reject_temperature 1.0 \
      --top_p 0.95 --top_k 50 \
      --shard_idx "$i" --num_shards 4 "$@" >>"$LOG" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] $tag failed" >&2; tail -60 "$LOG" >&2; exit 1; }
  echo "[ok] $tag"
}

# ---- stage 0: calibration, one pass per scorer -----------------------------
# The plain arm scores what it keeps. It decides nothing, so this is also the
# checker-blind reference the rejection arms are measured against.
echo "=== stage 0: per-scorer calibration on this dataset ==="
for CELL in $CELLS; do
  NAME=$(basename "$CELL")
  run_sharded "calib:$NAME" --arms plain --score_plain \
    --cell_dir "$CELL" --max_problems "$CALIB_PROBLEMS" \
    --out "$OUT_DIR/calib_$NAME/rollouts.jsonl"
done

echo "=== stage 0 gate: does the scorer discriminate on THIS data? ==="
OUT_DIR="$OUT_DIR" python - <<'PY'
import glob, json, os
import numpy as np
out = os.environ["OUT_DIR"]
ok = True
for d in sorted(glob.glob(f"{out}/calib_*")):
    rows = [json.loads(l) for f in glob.glob(f"{d}/*rollouts*.jsonl") for l in open(f)]
    rows = [r for r in rows if r.get("chosen_scores")]
    if not rows:
        print(f"[FAIL] {os.path.basename(d)}: no scored rollouts"); ok = False; continue
    s = np.array([max(r["chosen_scores"]) for r in rows])      # worst-step, the primary
    y = np.array([bool(r["correct"]) for r in rows])
    if y.all() or not y.any():
        print(f"[WARN] {os.path.basename(d)}: outcome has no variance"); continue
    from scipy.stats import rankdata
    rk = rankdata(-s); n1 = y.sum(); n0 = len(y) - n1
    auroc = (rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    qs = {q: float(np.quantile(np.concatenate([r["chosen_scores"] for r in rows]), q))
          for q in (0.50, 0.65, 0.80)}
    print(f"  {os.path.basename(d)}: n={len(rows)} AUROC(worst-step -> correct)={auroc:.3f} "
          f"taus={ {k: round(v,4) for k,v in qs.items()} }")
    json.dump({"auroc": auroc, "taus": qs}, open(f"{d}/calibration.json", "w"), indent=2)
    # A scorer that cannot rank trace outcomes on this data cannot usefully
    # reject its steps, and its rejection arms would only measure the retry loop.
    if auroc < 0.55:
        print(f"  [FAIL] AUROC {auroc:.3f} below 0.55: this scorer does not "
              f"transfer to this dataset"); ok = False
print("STAGE0 GATE:", "PASS" if ok else "FAIL")
if not ok:
    raise SystemExit(1)
PY

# ---- stage 1: the sweep ----------------------------------------------------
echo "=== stage 1: rejection sweep ==="
for CELL in $CELLS; do
  NAME=$(basename "$CELL")
  CAL="$OUT_DIR/calib_$NAME"
  for Q in $QUANTILES; do
    run_sharded "reject:$NAME:q$Q" --arms reject \
      --cell_dir "$CELL" --max_problems "$MAX_PROBLEMS" \
      --max_retries "$MAX_RETRIES" \
      --calibrate_from "$CAL/rollouts.jsonl" --reject_quantile "$Q" \
      --out "$OUT_DIR/reject_${NAME}_q${Q}/rollouts.jsonl"
  done
done

# The price tag: the identical retry loop with the accept decision made by a
# coin at the measured rate. §20.17 passed a per-step rate as a per-draw
# probability, giving 1.05 extra draws against reject's 0.54; solving
# p + p^2 = rate is what matches them.
echo "=== stage 2: blind control ==="
FIRST=$(echo $CELLS | awk '{print $1}')
BLIND_RATE="${BLIND_RATE:-0.54}"   # the reject arm's measured extra draws/step
run_sharded "blind" --arms reject_blind --cell_dir "$FIRST" \
  --max_problems "$MAX_PROBLEMS" --max_retries "$MAX_RETRIES" \
  --calibrate_from "$OUT_DIR/calib_$(basename $FIRST)/rollouts.jsonl" --reject_quantile 0.65 \
  --blind_target_extra_draws "$BLIND_RATE" \
  --out "$OUT_DIR/reject_blind/rollouts.jsonl"

echo "[$(date)] tts_online done"
