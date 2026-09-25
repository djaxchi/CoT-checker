#!/bin/bash
#SBATCH --job-name=tts_offline
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

# Job A of tts_roster_v1: the offline pool everything downstream reads.
#
# GSM8K test (1,319) and MATH500 (500), ten samples each, 4-shot CoT with a stop
# string, 2,048-token cap. Then the per-step signals that every offline
# selection rule needs: hidden states scored by the probe cells, and token
# logprobs for the DeepConf family.
#
# Runs beside slurm/tts_online_tamia.sh rather than before it: the online job
# calibrates its own thresholds on its own dataset, so neither waits on the
# other and the two together occupy 8 GPUs.
#
# Nothing here aggregates. Per-trace and per-step atoms are written and every
# derived number is computed downstream, so the explorer can add a selection
# rule later without regenerating (docs/tts_roster_v1_plan.md §2).

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/tts_roster_v1}"
SPLITS="${SPLITS:-$RUN_ROOT/splits}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
# fewshot for the Base policy; chat for Qwen/Qwen3-8B (instruct_arm_v1), which
# runs the non-thinking template and ignores --n_shot and --stop_strings.
PROMPT_STYLE="${PROMPT_STYLE:-fewshot}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
NUM_SHARDS="${NUM_SHARDS:-4}"
TOPK_CONF="${TOPK_CONF:-20}"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
LOG_DIR="$RUN_ROOT/logs"; mkdir -p "$RUN_ROOT" "$LOG_DIR"
LOG="$LOG_DIR/tts_offline-${SLURM_JOB_ID:-$$}.log"

SNAP="$HF_CACHE/hub/models--$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|--|g')/snapshots"
[[ -d "$SNAP" && -n "$(ls -A "$SNAP" 2>/dev/null)" ]] || {
  echo "[FATAL] no snapshot for $MODEL_NAME_OR_PATH" >&2; exit 2; }
for f in gsm8k_full math500_full; do
  [[ -f "$SPLITS/$f.jsonl" ]] || { echo "[FATAL] missing $SPLITS/$f.jsonl" >&2; exit 2; }
done

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-tts_offline}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
sets       : gsm8k_full (1319) + math500_full (500), x $N_SAMPLES samples
prompt     : 4-shot CoT, stop on delimiter, cap $MAX_NEW_TOKENS
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy

# ---- stage 1: generation, one shard per GPU --------------------------------
gen_set () {   # $1 = split stem, $2 = output stem
  local pids=()
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_onpolicy_steps.py \
      --fork_items "$SPLITS/$1.jsonl" --id_field problem_id \
      --out_dir "$RUN_ROOT" --stem "$2" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
      --model_dtype bfloat16 --run_name "$2" \
      --max_problems 0 --n_samples "$N_SAMPLES" \
      --temperature 1.0 --top_p 0.95 --top_k 50 \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --prompt_style "$PROMPT_STYLE" --n_shot 4 --stop_strings \
      --shard_idx "$i" --num_shards "$NUM_SHARDS" --force >>"$LOG" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] generation failed for $1" >&2
                            tail -60 "$LOG" >&2; exit 1; }
  echo "[ok] generated $2"
}

echo "=== stage 1: generation ==="
gen_set gsm8k_full   tts_gsm8k
gen_set math500_full tts_math500

# ---- stage 2: token logprobs for the DeepConf family -----------------------
echo "=== stage 2: token confidence ==="
for STEM in tts_gsm8k tts_math500; do
  pids=()
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/encode_token_confidence.py \
      --trajectories "$RUN_ROOT"/"$STEM".shard*_trajectories.jsonl \
      --out_dir "$RUN_ROOT" --stem "$STEM" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
      --model_dtype bfloat16 --topk "$TOPK_CONF" \
      --shard_idx "$i" --num_shards "$NUM_SHARDS" --force >>"$LOG" 2>&1 &
    pids+=($!)
  done
  fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] confidence failed for $STEM" >&2
                            tail -60 "$LOG" >&2; exit 1; }
  echo "[ok] confidence $STEM"
done

# ---- stage 3: gates --------------------------------------------------------
echo "=== stage 3: gates ==="
RUN_ROOT="$RUN_ROOT" python - <<'PY'
import glob, json, os, statistics as st
root = os.environ["RUN_ROOT"]
ok = True
summary = {}
for stem, target in (("tts_gsm8k", 0.8984), ("tts_math500", 0.6080)):
    rows = [json.loads(l) for f in glob.glob(f"{root}/{stem}.shard*_trajectories.jsonl")
            for l in open(f)]
    if not rows:
        print(f"[FAIL] {stem}: no trajectories"); ok = False; continue
    cap = sum(r.get("hit_token_cap", False) for r in rows) / len(rows)
    acc = sum(r["correct"] for r in rows) / len(rows)
    grad = sum(r["gradeable"] for r in rows) / len(rows)
    toks = sorted(r.get("n_gen_tokens", 0) for r in rows)
    boxed = sum("\\boxed{" in r["solution"] for r in rows) / len(rows)
    summary[stem] = {"n": len(rows), "hit_cap": cap, "acc": acc, "gradeable": grad,
                     "boxed": boxed, "tok_med": st.median(toks),
                     "tok_p95": toks[int(.95 * len(toks)) - 1], "tok_max": toks[-1]}
    print(f"  {stem}: n={len(rows)} hit_cap={cap:.2%} acc={acc:.3f} "
          f"(published {target}) gradeable={grad:.1%} boxed={boxed:.1%} "
          f"tokens med={st.median(toks):.0f} p95={toks[int(.95*len(toks))-1]} max={toks[-1]}")
    if cap > 0.02:
        print(f"  [FAIL] {stem} truncation {cap:.2%} above the 2% gate"); ok = False
# Span coverage is gated inside encode_token_confidence, which exits nonzero
# below 0.90, so reaching here means it passed.
for f in sorted(glob.glob(f"{root}/tts_*_conf_manifest.json")):
    m = json.load(open(f))
    summary.setdefault("span", []).append(m.get("span_coverage_rate"))
json.dump(summary, open(f"{root}/offline_gates.json", "w"), indent=2)
print("GATE:", "PASS" if ok else "FAIL")
PY

echo "[$(date)] tts_offline done"
