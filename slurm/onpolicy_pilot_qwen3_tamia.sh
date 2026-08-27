#!/bin/bash
#SBATCH --job-name=onpolicy_pilot
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# Pilot for the on-policy arm: can Qwen3-8B-Base generate usable step-segmented
# CoT for problems the off-policy grid already covers?
#
# The whole grid so far is OFF-policy: PRM800K solutions were written by a GPT-4
# fine-tune and we re-encode them. Every comparable paper reads the states of the
# model that generated the text. Before building the on-policy arm, this checks
# the one thing everything downstream depends on.
#
# Go/no-go, in order of what would kill the arm:
#   1. does a BASE model produce parseable multi-step solutions from this prompt
#   2. is trajectory accuracy inside a usable band -- near 0% leaves no correct
#      steps to learn from, near 100% leaves no incorrect ones
#   3. are steps comparable in length to PRM800K's 38.8 tokens, so the
#      representations mean the same thing
#
# Problems come from the PRM800K TEST split, so they are held out of probe
# training and are problems the off-policy grid was scored on.
# NO INTERNET on compute nodes: weights must already be cached.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/qwen3_onpolicy_pilot}"
PROBLEMS="${PROBLEMS:-$SCRATCH/cot_mech/dense_full_7b_v1/data/prm800k_test_2k.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
MAX_PROBLEMS="${MAX_PROBLEMS:-48}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
STEM="${STEM:-onpolicy_pilot}"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_ROOT"

SNAP="$HF_CACHE/hub/models--$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|--|g')/snapshots"
[[ -d "$SNAP" && -n "$(ls -A "$SNAP" 2>/dev/null)" ]] || {
  echo "[FATAL] no local snapshot for $MODEL_NAME_OR_PATH; download on the login node" >&2
  exit 2; }

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job       : ${SLURM_JOB_NAME:-onpolicy_pilot}  id: ${SLURM_JOB_ID:-N/A}
model     : $MODEL_NAME_OR_PATH (base, offline)
problems  : $PROBLEMS  (max $MAX_PROBLEMS)
sampling  : $N_SAMPLES per problem, T=$TEMPERATURE, max_new=$MAX_NEW_TOKENS
out       : $RUN_ROOT
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy

python scripts/generate_onpolicy_steps.py \
  --fork_items "$PROBLEMS" --id_field problem_id \
  --out_dir "$RUN_ROOT" --stem "$STEM" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
  --model_dtype bfloat16 \
  --run_name "onpolicy_pilot_qwen3" \
  --max_problems "$MAX_PROBLEMS" --n_samples "$N_SAMPLES" \
  --temperature "$TEMPERATURE" --max_new_tokens "$MAX_NEW_TOKENS" --force

echo "=== pilot verdict ==="
python - <<PY
import json, numpy as np
from pathlib import Path
R = Path("$RUN_ROOT")
tr = [json.loads(l) for l in (R / "${STEM}_trajectories.jsonl").read_text().splitlines() if l.strip()]
it = [json.loads(l) for l in (R / "${STEM}_items.jsonl").read_text().splitlines() if l.strip()]
acc = np.mean([t["correct"] for t in tr]) if tr else 0.0
steps = np.array([t["n_steps"] for t in it if t["step_idx"] == 0]) if it else np.array([0])
words = np.array([len(t["candidate_step"].split()) for t in it]) if it else np.array([0])
print(f"trajectories {len(tr)}  steps {len(it)}")
print(f"trajectory accuracy      {acc:.3f}   (want 0.20-0.80: both classes present)")
print(f"steps per solution       median {np.median(steps):.0f}  mean {steps.mean():.1f}  (want >= 3)")
print(f"step length (words)      median {np.median(words):.0f}  mean {words.mean():.1f}")
print(f"single-step solutions    {(steps <= 1).mean():.3f}   (want low: segmentation works)")
ok = (0.10 < acc < 0.95) and np.median(steps) >= 3 and (steps <= 1).mean() < 0.3
print()
print("VERDICT:", "GO" if ok else "NO-GO -- see which criterion failed above")
PY
echo "[$(date)] onpolicy_pilot done"
