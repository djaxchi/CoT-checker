#!/bin/bash
#SBATCH --job-name=tts_sota
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Rescore the tts_roster_v1 pool with the verifiers that should have run the
# first time, plus the PRM the field would actually deploy.
#
# NOTHING IS REGENERATED. The 18,188 trajectories, their token counts and their
# confidence statistics are already on disk. This is four scoring passes over
# saved text, so the expensive half of the original study is not repeated.
#
# WHY THESE CELLS. The first run scored one cell, transformer M, ranked 8th on
# the leaderboard. Checker only requires rep == "step_tokens"; within that,
# transformer L (ProcessBench F1 0.566) and attn_query (0.558, 8,193 parameters)
# were both trained and eligible and neither was used. The M cell was the default
# in the launcher, inherited from the rejection-sampling run, and never revisited.
# M stays in the list so the published curves remain comparable.
#
# attn_query is the interesting one: if 8,193 parameters match a 7B PRM on the
# same draws, that is the result, and it cannot be claimed until both are scored
# on one pool.
#
# COST. One backbone forward per step per probe cell. Checker is the gated
# scoring path and is not refactored to share that pass across cells: a change
# there would be exactly the kind that makes every number wrong without looking
# wrong. Three probe passes plus one 7B PRM prefill pass.
#
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/reprobe_v1}"
TTS_ROOT="${TTS_ROOT:-$SCRATCH/cot_mech/tts_roster_v1}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
PRM_NAME_OR_PATH="${PRM_NAME_OR_PATH:-Qwen/Qwen2.5-Math-PRM-7B}"
REP_ROOT="${REP_ROOT:-$RUN_ROOT/repstore/onpolicy_train_spans}"
LAYER="${LAYER:-35}"
NUM_SHARDS="${NUM_SHARDS:-4}"
STEMS="${STEMS:-tts_gsm8k tts_math500}"

# Probe cells are opt-in. The default run is the PRM alone, which is ~18K short
# prefills through a 7B model and fits the 1h walltime; each probe cell adds a
# full backbone pass. To add them, submit with e.g.
#   CELLS="$RUN_ROOT/cells/step_tokens__attn_query__seed42" sbatch --time=03:00:00 ...
CELLS="${CELLS:-}"

cd "$PROJECT_ROOT"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Same throwaway environment as slurm/tts_score_cells_tamia.sh (job 465706),
# which is the scoring job known to work on this pool.
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
# transformers < 5: the PRM's remote code breaks on 5.x (checked on the login
# node with a 1-layer copy of its config; 4.57.6 runs it with use_cache=False).
pip install --no-index torch "transformers<5" numpy sympy 2>&1 | tail -1

# Fail in seconds, not after the first shard has loaded, if an input is missing.
for STEM in $STEMS; do
  ls "$TTS_ROOT"/${STEM}.shard*_trajectories.jsonl >/dev/null 2>&1 \
    || { echo "[FATAL] no trajectories for $STEM under $TTS_ROOT" >&2; exit 2; }
  ls "$TTS_ROOT"/scores/${STEM}__step_tokens__transformer_d256_l2_f1024_h4__seed42.shard*.jsonl >/dev/null 2>&1 \
    || { echo "[FATAL] no probe scores for $STEM: the segmentation gate has nothing to check" >&2; exit 2; }
done
python - <<PY || { echo "[FATAL] $PRM_NAME_OR_PATH is not loadable offline from $HF_CACHE" >&2; exit 2; }
from transformers import AutoConfig
AutoConfig.from_pretrained("$PRM_NAME_OR_PATH", trust_remote_code=True, local_files_only=True)
print("[preflight] PRM config found")
PY

mkdir -p "$TTS_ROOT/scores" "$RUN_ROOT/logs"
LOG="$RUN_ROOT/logs/tts_sota-${SLURM_JOB_ID:-local}.log"
git -C "$PROJECT_ROOT" rev-parse --short HEAD | tee -a "$LOG"

run_sharded () {  # run_sharded <stem> <out_stem> <cmd...>
  local stem="$1" out="$2"; shift 2
  local pids=()
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i "$@" \
      --trajectories "$TTS_ROOT"/${stem}.shard*_trajectories.jsonl \
      --shard_idx "$i" --num_shards "$NUM_SHARDS" \
      --out "$TTS_ROOT/scores/${out}.shard${i}.jsonl" >>"$LOG" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] $out failed" >&2; tail -60 "$LOG" >&2; exit 1; }
  echo "[ok] $out"
}

echo "=== 1. the probe cells that should have run the first time ==="
for CELL in $CELLS; do
  NAME=$(basename "$CELL" | sed 's/__seed42$//')
  for STEM in $STEMS; do
    echo "--- $STEM / $NAME ---" | tee -a "$LOG"
    run_sharded "$STEM" "${STEM}__${NAME}__seed42" \
      python scripts/onpolicy/score_traces_with_cell.py \
        --cell_dir "$CELL" --layer "$LAYER" \
        --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
        --prm_store "$REP_ROOT" --stats_cache "$RUN_ROOT/stats_cache"
  done
done

echo
echo "=== 2. the PRM, with both gates armed ==="
# --verify_against points at the cell scores already on disk, so the PRM has to
# segment every trace exactly as the probe did or the job aborts without writing.
for STEM in $STEMS; do
  # Every probe shard: PRM shards and probe shards stride the pool differently.
  REFS=("$TTS_ROOT"/scores/${STEM}__step_tokens__transformer_d256_l2_f1024_h4__seed42.shard*.jsonl)
  echo "--- $STEM / PRM (gate against ${#REFS[@]} probe shards) ---" | tee -a "$LOG"
  run_sharded "$STEM" "${STEM}__prm_qwen25_math_7b" \
    python scripts/onpolicy/score_traces_with_prm.py \
      --prm_name_or_path "$PRM_NAME_OR_PATH" --local_files_only \
      --verify_against "${REFS[@]}"
done
grep -E "\[prm\] (orientation|segmentation|shard)" "$LOG" || true

# Worst-step AUROC per scorer. Ranks by hand: no scipy in this venv (job 465706).
TTS_ROOT="$TTS_ROOT" python - <<'PY'
import glob, json, os
import numpy as np
root = os.environ["TTS_ROOT"]
bases = sorted({f.rsplit(".shard", 1)[0] for f in glob.glob(f"{root}/scores/*.shard*.jsonl")})
for base in bases:
    rows = [json.loads(l) for g in glob.glob(base + ".shard*.jsonl") for l in open(g)]
    worst = np.array([max(r["scores"]) for r in rows])
    y = np.array([bool(r["correct"]) for r in rows])
    order = np.argsort(-worst, kind="mergesort")
    rk = np.empty(len(worst)); rk[order] = np.arange(1, len(worst) + 1)
    n1, n0 = y.sum(), (~y).sum()
    auroc = (rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    print(f"  {os.path.basename(base)}: n={len(rows)} AUROC(worst-step -> correct)={auroc:.3f}")
PY

echo "[done] $TTS_ROOT/scores"
echo "Next, on the laptop, no GPU needed:"
echo "  python scripts/analysis/tts_build_frontier.py --run_root <staged> \\"
echo "    --n_orders 32 --out results/tts_roster_v1/frontier_v3.json"
