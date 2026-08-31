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
K_ROLLOUTS="${K_ROLLOUTS:-16}"
MAX_FORKS="${MAX_FORKS:-300}"
MAX_TRACES="${MAX_TRACES:-0}"
N_CORRECT_AUDIT="${N_CORRECT_AUDIT:-200}"
RULE="${RULE:-zero}"
NUM_SHARDS="${NUM_SHARDS:-4}"
OUT="$RUN_ROOT/rollout"
# Two conditions, because the first run (job 433686) showed they are different
# questions. Among the forks the rollouts separate at all, the human-rated wrong
# step must have the lower value at least MIN_WIN_RATE of the time: that is the
# labeller's discrimination. And enough forks must be decided for the first
# number to mean anything: that is the run's power, set by K and by how often the
# model can solve the problem from either branch, not by the labeller.
#
# At K=4 the first run tied on 61.5% of 200 forks, nearly all at zero against
# zero, and scored 0.766 among the 77 it decided (sign test p ~ 3e-7). The signal
# was there and the resolution was not, so K goes to 16.
MIN_WIN_RATE="${MIN_WIN_RATE:-0.60}"
MIN_DECIDED="${MIN_DECIDED:-0.35}"

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
phase 1    : $MAX_FORKS matched forks, K=$K_ROLLOUTS
             gate: decided >= $MIN_DECIDED and win rate among decided >= $MIN_WIN_RATE
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
import sys
sys.path.insert(0, "$PROJECT_ROOT")
from scripts.onpolicy.rollout_labels import sign_test_p
vp = np.array([r["v_pos"] for r in rows]); vn = np.array([r["v_neg"] for r in rows])
dec = vp != vn
n_dec = int(dec.sum()); wins = int((vn[dec] < vp[dec]).sum())
summary = {
    "n_pairs": len(rows), "n_decided": n_dec,
    "decided_fraction": n_dec / max(1, len(rows)),
    "win_rate_among_decided": (wins / n_dec) if n_dec else float("nan"),
    "sign_test_p": sign_test_p(wins, n_dec),
    "win_rate_all_pairs": float((vn < vp).mean()),
    "ties": float((~dec).mean()),
    "ties_both_zero": float(((vp == 0) & (vn == 0)).mean()),
    "mean_value_positive": float(vp.mean()), "mean_value_negative": float(vn.mean()),
}
json.dump(summary, open("$OUT/cert_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print()
print(f"{len(rows)} matched forks, {n_dec} decided by the rollouts "
      f"({summary['decided_fraction']:.3f}); {summary['ties_both_zero']:.3f} tied "
      f"at zero against zero, which says the model cannot solve those problems "
      f"from either branch and not that the labeller failed.")
print(f"Among the decided, the step humans rated WRONG has the lower value "
      f"{summary['win_rate_among_decided']:.3f} of the time "
      f"(sign test p = {summary['sign_test_p']:.2e}). Chance is 0.500.")
PY

GATE=$(python -c "
import json
s = json.load(open('$OUT/cert_summary.json'))
ok = s['win_rate_among_decided'] >= $MIN_WIN_RATE and s['decided_fraction'] >= $MIN_DECIDED
print(1 if ok else 0)")
if [[ "$GATE" != "1" ]]; then
  echo "[GATE FAILED] see cert_summary.json. A low win rate among decided pairs"
  echo "              means the labeller is not reading correctness. A low"
  echo "              decided fraction means this run had no resolution: raise"
  echo "              K_ROLLOUTS rather than concluding anything."
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
