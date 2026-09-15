#!/bin/bash
#SBATCH --job-name=tts_smoke
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out

# The gate that decides whether tts_roster_v1 launches tonight.
#
# The previous pool ran a zero-shot prompt that showed the model nothing about
# finishing, and 21.2% of its traces ran to the 768-token cap (REPORT.md
# §20.16). Those traces are correct 5.5% of the time, the grader's fallbacks
# still parse an answer from them, and they handed every cheap baseline a
# correctness signal that was really a budget signal.
#
# Raising the cap alone is the expensive fix: its cost lands entirely on traces
# that never terminate. Few-shot prompting is the cheap one, and this measures
# whether it works before 12 node-hours are committed to it.
#
# PASS: truncation below 2% on both sets, and single-sample accuracy within
# 0.05 of the Qwen3 report's 89.84 GSM8K / 60.80 MATH at 4-shot CoT.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/tts_roster_v1}"
SPLITS="${SPLITS:-$RUN_ROOT/splits}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
N_SAMPLES="${N_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_SHARDS="${NUM_SHARDS:-4}"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
LOG_DIR="$RUN_ROOT/logs"; mkdir -p "$RUN_ROOT" "$LOG_DIR"
LOG_FILE="$LOG_DIR/tts_smoke-${SLURM_JOB_ID:-$$}.log"

SNAP="$HF_CACHE/hub/models--$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|--|g')/snapshots"
[[ -d "$SNAP" && -n "$(ls -A "$SNAP" 2>/dev/null)" ]] || {
  echo "[FATAL] no local snapshot for $MODEL_NAME_OR_PATH" >&2; exit 2; }
[[ -f "$SPLITS/smoke.jsonl" ]] || {
  echo "[FATAL] no $SPLITS/smoke.jsonl; run build_tts_splits.py on the login node" >&2
  exit 2; }

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-tts_smoke}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH
problems   : $SPLITS/smoke.jsonl  x $N_SAMPLES samples
prompt     : 4-shot CoT, stop on the next-problem delimiter
cap        : $MAX_NEW_TOKENS tokens (was 768)
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy

pids=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/generate_onpolicy_steps.py \
    --fork_items "$SPLITS/smoke.jsonl" --id_field problem_id \
    --out_dir "$RUN_ROOT" --stem tts_smoke \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --model_dtype bfloat16 --run_name tts_smoke \
    --max_problems 0 --n_samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE" --top_p 0.95 --top_k 50 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt_style fewshot --n_shot 4 --stop_strings \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" --force >>"$LOG_FILE" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
if [[ "$fail" == "1" ]]; then tail -60 "$LOG_FILE" >&2; exit 1; fi

echo "=== SMOKE GATE ==="
RUN_ROOT="$RUN_ROOT" python - <<'PY'
import glob, json, os, statistics as st
root = os.environ["RUN_ROOT"]
rows = [json.loads(l) for f in glob.glob(f"{root}/tts_smoke.shard*_trajectories.jsonl")
        for l in open(f)]
print(f"trajectories: {len(rows)}")
ok = True
for ds in sorted({r.get("dataset", "?") for r in rows}):
    sub = [r for r in rows if r.get("dataset") == ds]
    cap = sum(r.get("hit_token_cap", False) for r in sub) / len(sub)
    acc = sum(r["correct"] for r in sub) / len(sub)
    toks = [r.get("n_gen_tokens", 0) for r in sub]
    boxed = sum("\\boxed{" in r["solution"] for r in sub) / len(sub)
    target = {"gsm8k": 0.8984, "math": 0.6080}.get(ds)
    print(f"  {ds:8s} n={len(sub):4d}  hit_cap={cap:.2%}  acc={acc:.3f}"
          f"  (published {target})  boxed={boxed:.1%}"
          f"  tokens med={st.median(toks):.0f} p95={sorted(toks)[int(.95*len(toks))-1]:.0f}"
          f" max={max(toks)}")
    if cap > 0.02:
        print(f"  [FAIL] {ds} truncation {cap:.2%} above the 2% gate"); ok = False
    if target is not None and abs(acc - target) > 0.05:
        print(f"  [WARN] {ds} accuracy {acc:.3f} is {abs(acc-target):.3f} from "
              f"published {target}; sampling at T=1.0 sits below greedy few-shot, "
              f"so judge this against the plain arm rather than failing on it")
print("GATE:", "PASS" if ok else "FAIL")
PY
echo "[$(date)] tts_smoke done"
