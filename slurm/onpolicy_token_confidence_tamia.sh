#!/bin/bash
#SBATCH --job-name=onpolicy_conf
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Phase 1 of onpolicy_tiebreak_v2: recover the token logprobs that REPORT.md
# §20.11 recorded as unobtainable, so the tie-break claim can finally be raced
# against DeepConf-style token confidence instead of only against length rules.
#
# No regeneration. The trajectory text is on disk and src/onpolicy/prompts.py
# rebuilds the exact sampling prompt, so one teacher-forced forward pass per
# trajectory recovers the model's distribution at every generated position.
# That turns a 64K-trajectory regeneration into ~2,900 single forward passes.
#
# The job fails loudly if step-span coverage drops below 90%: misaligned spans
# would give per-step confidences that look plausible and are wrong.
#
# NO INTERNET on compute nodes: Qwen3-8B-Base must already be in $HF_CACHE.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/onpolicy_v1}"
TRAJ_GLOB="${TRAJ_GLOB:-$RUN_ROOT/onpolicy_stage1.shard*_trajectories.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
STEM="${STEM:-onpolicy_stage1}"
TOPK="${TOPK:-20}"
MAX_TRACES="${MAX_TRACES:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
LOG_DIR="$RUN_ROOT/logs"; mkdir -p "$RUN_ROOT" "$LOG_DIR"
LOG_FILE="$LOG_DIR/onpolicy_conf-${SLURM_JOB_ID:-$$}.log"

SNAP="$HF_CACHE/hub/models--$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|--|g')/snapshots"
[[ -d "$SNAP" && -n "$(ls -A "$SNAP" 2>/dev/null)" ]] || {
  echo "[FATAL] no local snapshot for $MODEL_NAME_OR_PATH" >&2; exit 2; }
# shellcheck disable=SC2086
[[ -n "$(ls $TRAJ_GLOB 2>/dev/null)" ]] || {
  echo "[FATAL] no trajectories matching $TRAJ_GLOB" >&2; exit 2; }

cd "$PROJECT_ROOT"
cat <<BANNER
================================================================
job        : ${SLURM_JOB_NAME:-onpolicy_conf}  id: ${SLURM_JOB_ID:-N/A}
git_commit : $(git rev-parse HEAD 2>/dev/null || echo unknown)
model      : $MODEL_NAME_OR_PATH (offline)
trajectories: $TRAJ_GLOB
confidence : DeepConf Eq 2 with k=$TOPK
shards     : $NUM_SHARDS on ${SLURM_GPUS_ON_NODE:-4} GPUs
out        : $RUN_ROOT/${STEM}.shardNN_conf.{npz,jsonl}
================================================================
BANNER

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy

pids=(); tags=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  echo "[launch] shard $i -> GPU $i" | tee -a "$LOG_FILE"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/encode_token_confidence.py \
    --trajectories $TRAJ_GLOB \
    --out_dir "$RUN_ROOT" --stem "$STEM" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
    --model_dtype bfloat16 --topk "$TOPK" --max_traces "$MAX_TRACES" \
    --shard_idx "$i" --num_shards "$NUM_SHARDS" --force >>"$LOG_FILE" 2>&1 &
  pids+=($!); tags+=("shard_$i")
done

fail=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[ok] ${tags[$j]}"; else echo "[FAIL] ${tags[$j]}"; fail=1; fi
done
if [[ "$fail" == "1" ]]; then tail -60 "$LOG_FILE" >&2; exit 1; fi

echo "=== span coverage across shards ==="
python - <<'PY'
import glob, json, os
root = os.environ.get("RUN_ROOT", os.path.expandvars("$SCRATCH/cot_mech/onpolicy_v1"))
stem = os.environ.get("STEM", "onpolicy_stage1")
ok = tot = n = 0
for p in sorted(glob.glob(f"{root}/{stem}.shard*_conf_manifest.json")):
    m = json.load(open(p))
    ok += m["span_coverage_pass"]; tot += m["span_coverage_total"]; n += m["n_traces"]
print(f"traces={n} span_coverage={ok}/{tot} = {ok/tot:.4f}" if tot else "no manifests")
PY

echo "[$(date)] onpolicy_conf done"
