#!/bin/bash
#SBATCH --job-name=onpolicy_rollout
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Stage 2b: label the first wrong step by rollout instead of by judge, and
# certify the labeller against human annotations before using it.
#
# Why this arm exists: the judge bake-off (job 433640) put three local judges at
# F1_PB 0.42-0.44 on human-labelled ProcessBench traces with Acc_error 0.29,
# against 0.566 for the best representation cell on the same metric. Labels
# noisier than the thing being measured cannot carry a rank claim.
#
# PHASE 1 is a gate, and runs first. On PRM800K matched forks (one prefix, one
# step humans rated +1, one they rated -1), roll out from both continuations and
# ask whether the value is lower after the step humans called wrong. Same prefix,
# same problem, so nothing but the step differs. If the labeller cannot separate
# a human-annotated wrong step from a right one at the same fork, its labels are
# not worth generating and phase 2 is skipped.
#
# PHASE 2 labels the on-policy trajectories, and audits itself: correct
# trajectories are labelled -1 from the grader either way, and how often the rule
# would have fired on them is the false-alarm rate on the distribution that
# matters, measured without a single human label.
#
# Sharded four ways in-node. NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/onpolicy_v1}"
FORKS="${FORKS:-/scratch/d/dchikhi/cot-checker/transition_operator/forks.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
STEM="${STEM:-onpolicy_stage1}"
K_ROLLOUTS="${K_ROLLOUTS:-4}"
MAX_FORKS="${MAX_FORKS:-200}"
MAX_TRACES="${MAX_TRACES:-0}"
N_CORRECT_AUDIT="${N_CORRECT_AUDIT:-200}"
RULE="${RULE:-zero}"
NUM_SHARDS="${NUM_SHARDS:-4}"
OUT="$RUN_ROOT/rollout"
# The labeller has to separate a human-rated wrong step from a right one at the
# same fork by this much, or its labels are noise dressed as supervision.
MIN_WIN_RATE="${MIN_WIN_RATE:-0.60}"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT" "$RUN_ROOT/logs"
LOG_FILE="$RUN_ROOT/logs/onpolicy_rollout-${SLURM_JOB_ID:-$$}.log"

SNAP="$HF_CACHE/hub/models--$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|--|g')/snapshots"
[[ -d "$SNAP" && -n "$(ls -A "$SNAP" 2>/dev/null)" ]] || {
  echo "[FATAL] no local snapshot for $MODEL_NAME_OR_PATH" >&2; exit 2; }
[[ -f "$FORKS" ]] || { echo "[FATAL] no fork file at $FORKS" >&2; exit 2; }

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-onpolicy_rollout}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH (offline)
phase 1    : $MAX_FORKS matched forks, K=$K_ROLLOUTS   gate: win rate >= $MIN_WIN_RATE
phase 2    : $RUN_ROOT/$STEM.shard*_trajectories.jsonl, rule=$RULE
out        : $OUT
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy

run_sharded () {   # $1 = tag, rest = args
  local tag="$1"; shift
  local pids=() tags=()
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/rollout_labels.py "$@" \
      --shard_idx "$i" --num_shards "$NUM_SHARDS" \
      --out "$OUT/${tag}.shard${i}.jsonl" \
      --report "$OUT/${tag}.shard${i}_report.json" >>"$LOG_FILE" 2>&1 &
    pids+=($!); tags+=("${tag}_shard_$i")
  done
  local fail=0
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "[ok] ${tags[$j]}"; else echo "[FAIL] ${tags[$j]}"; fail=1; fi
  done
  [[ "$fail" == "0" ]] || { tail -40 "$LOG_FILE" >&2; return 1; }
}

echo "=== phase 1: certify against human fork annotations ==="
run_sharded cert \
  --certify_forks "$FORKS" --max_forks "$MAX_FORKS" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --model_dtype bfloat16 --k_rollouts "$K_ROLLOUTS"

python - <<PY | tee -a "$LOG_FILE"
import glob, json
import numpy as np
rows = [json.loads(l) for f in sorted(glob.glob("$OUT/cert.shard*.jsonl"))
        for l in open(f) if l.strip()]
vp = np.array([r["v_pos"] for r in rows]); vn = np.array([r["v_neg"] for r in rows])
dec = vp != vn
win = float((vn < vp).mean())
summary = {
    "n_pairs": len(rows), "win_rate": win,
    "win_rate_among_decided": float((vn[dec] < vp[dec]).mean()) if dec.any() else float("nan"),
    "ties": float((~dec).mean()),
    "mean_value_positive": float(vp.mean()), "mean_value_negative": float(vn.mean()),
}
json.dump(summary, open("$OUT/cert_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print()
print(f"The step humans rated WRONG has the lower rollout value in {win:.3f} of "
      f"{len(rows)} matched forks ({summary['win_rate_among_decided']:.3f} of the "
      f"{int(dec.sum())} the rollouts separate at all; {summary['ties']:.3f} tie).")
print("Chance is 0.500. A labeller at chance here is not reading correctness.")
PY

GATE=$(python -c "import json;print(1 if json.load(open('$OUT/cert_summary.json'))['win_rate'] >= $MIN_WIN_RATE else 0)")
if [[ "$GATE" != "1" ]]; then
  echo "[GATE FAILED] the rollout labeller does not separate human-rated wrong"
  echo "              steps from right ones at the same fork. Not generating"
  echo "              labels from it. The judge arm stands as the alternative."
  exit 3
fi

echo "=== phase 2: label the on-policy trajectories ==="
run_sharded labels \
  --trajectories "$RUN_ROOT"/"$STEM".shard*_trajectories.jsonl \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --model_dtype bfloat16 --k_rollouts "$K_ROLLOUTS" --rule "$RULE" \
  --max_traces "$MAX_TRACES" --n_correct_audit "$N_CORRECT_AUDIT"

cat "$OUT"/labels.shard*.jsonl > "$OUT/labels.jsonl"
wc -l "$OUT/labels.jsonl"

python scripts/onpolicy/build_pb_traces.py \
  --trajectories "$RUN_ROOT"/"$STEM".shard*_trajectories.jsonl \
  --labels "$OUT/labels.jsonl" \
  --out_dir "$RUN_ROOT" --stem "$STEM" --force | tee -a "$LOG_FILE"

echo "[$(date)] onpolicy_rollout done"
