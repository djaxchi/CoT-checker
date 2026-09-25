#!/bin/bash
#SBATCH --job-name=tts_genstates
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Score the tts_roster_v1 pool from the states the sampler computed, not from a
# reread under the verifier template. This is the only version of the verifier
# whose cost is close to zero in deployment, and it had not been measured on
# this pool (the published curves came from the reread, job 465706).
#
# One teacher-forced Qwen3-8B-Base pass per trajectory reconstructs those states.
# The script times the backbone and the head separately so the report can say
# what deployment would actually pay (the head only).
#
# Same environment, cell and rescaling statistics as slurm/tts_score_cells_tamia.sh.
# NO INTERNET on compute nodes.

set -euo pipefail
module load StdEnv/2023 python/3.12 gcc arrow/24.0.0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CoT-checker}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/cot_mech/tts_roster_v1}"
REPROBE_ROOT="${REPROBE_ROOT:-$SCRATCH/cot_mech/reprobe_v1}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B-Base}"
HF_CACHE="${HF_CACHE:-/project/aip-azouaq/$USER/hf_cache}"
REP_ROOT="${REP_ROOT:-$REPROBE_ROOT/repstore/onpolicy_train_spans}"
STATS_CACHE="${STATS_CACHE:-$REPROBE_ROOT/stats_cache}"
LAYER="${LAYER:-35}"
CELL="${CELL:-$REPROBE_ROOT/cells/step_tokens__transformer_d256_l2_f1024_h4__seed42}"
NAME="$(basename "$CELL")__gen"

export HF_HOME="$HF_CACHE"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
LOG_DIR="$RUN_ROOT/logs"; mkdir -p "$LOG_DIR" "$RUN_ROOT/scores"
LOG="$LOG_DIR/tts_genstates-${SLURM_JOB_ID:-$$}.log"

for STEM in tts_gsm8k tts_math500; do
  ls "$RUN_ROOT"/"$STEM".shard*_trajectories.jsonl >/dev/null 2>&1 \
    || { echo "[FATAL] no trajectories for $STEM" >&2; exit 2; }
done
[[ -f "$CELL/model.pt" ]] || { echo "[FATAL] no cell at $CELL" >&2; exit 2; }

cd "$PROJECT_ROOT"
echo "git_commit: $(git rev-parse --short HEAD)" | tee -a "$LOG"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy pyarrow pandas 2>&1 | tail -1

for STEM in tts_gsm8k tts_math500; do
  echo "=== $NAME on $STEM ===" | tee -a "$LOG"
  pids=()
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/score_traces_generation_states.py \
      --trajectories "$RUN_ROOT"/"$STEM".shard*_trajectories.jsonl \
      --cell_dir "$CELL" --layer "$LAYER" \
      --prm_store "$REP_ROOT" --stats_cache "$STATS_CACHE" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
      --model_dtype bfloat16 \
      --out "$RUN_ROOT/scores/${STEM}__${NAME}.shard0${i}.jsonl" \
      --shard_idx "$i" --num_shards 4 >>"$LOG" 2>&1 &
    pids+=($!)
  done
  fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[FATAL] $NAME/$STEM" >&2; tail -60 "$LOG" >&2; exit 1; }
  echo "[ok] $NAME on $STEM" | tee -a "$LOG"
done
grep -E "\[gen\] span coverage" "$LOG" || true

# Worst-step AUROC, reread against generation states. Ranks by hand: no scipy.
RUN_ROOT="$RUN_ROOT" python - <<'PY'
import glob, json, os
import numpy as np
root = os.environ["RUN_ROOT"]
bases = sorted({f.rsplit(".shard", 1)[0] for f in glob.glob(f"{root}/scores/*.shard*.jsonl")})
for base in bases:
    rows = [json.loads(l) for g in glob.glob(base + ".shard*.jsonl") for l in open(g)]
    worst = np.array([max(r["scores"]) for r in rows])
    y = np.array([bool(r["correct"]) for r in rows])
    order = np.argsort(-worst, kind="mergesort")
    rk = np.empty(len(worst)); rk[order] = np.arange(1, len(worst) + 1)
    n1, n0 = y.sum(), (~y).sum()
    print(f"  {os.path.basename(base)}: n={len(rows)} "
          f"AUROC(worst-step -> correct)={(rk[y].sum() - n1*(n1+1)/2)/(n1*n0):.3f}")
PY
echo "[$(date)] tts_genstates done"
