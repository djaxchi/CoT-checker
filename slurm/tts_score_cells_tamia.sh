#!/bin/bash
#SBATCH --job-name=tts_score
#SBATCH --account=aip-azouaq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Job A gives traces and token logprobs. Without this, tomorrow's offline
# comparison has majority voting, the DeepConf family, mean logprob, length and
# has_boxed, and no hidden-state verifier, which is the one thing the study is
# about.
#
# Scores every step of the generated pool with trained cells, through the same
# Checker the online arm uses, so offline and online numbers come from one
# implementation rather than two that can drift.
#
# Submitted with --dependency=afterok on the generation job.

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
# Offline scoring is a batch pass, so the pooled readout that Checker refuses to
# drive online is fine here. This is where the representation comparison lives.
CELLS="${CELLS:-$REPROBE_ROOT/cells/step_tokens__transformer_d256_l2_f1024_h4__seed42}"

export HF_HOME="$HF_CACHE" TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
LOG_DIR="$RUN_ROOT/logs"; mkdir -p "$RUN_ROOT" "$LOG_DIR"
LOG="$LOG_DIR/tts_score-${SLURM_JOB_ID:-$$}.log"

[[ -d "$REP_ROOT" ]] || { echo "[FATAL] no repstore at $REP_ROOT" >&2; exit 2; }
for STEM in tts_gsm8k tts_math500; do
  ls "$RUN_ROOT"/"$STEM".shard*_trajectories.jsonl >/dev/null 2>&1 || {
    echo "[FATAL] no trajectories for $STEM; did the generation job finish?" >&2
    exit 2; }
done

cd "$PROJECT_ROOT"
echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers numpy sympy pyarrow pandas

for CELL in $CELLS; do
  NAME=$(basename "$CELL")
  for STEM in tts_gsm8k tts_math500; do
    echo "=== $NAME on $STEM ==="
    pids=()
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python scripts/onpolicy/score_traces_with_cell.py \
        --trajectories "$RUN_ROOT"/"$STEM".shard*_trajectories.jsonl \
        --cell_dir "$CELL" --layer "$LAYER" \
        --prm_store "$REP_ROOT" --stats_cache "$STATS_CACHE" \
        --model_name_or_path "$MODEL_NAME_OR_PATH" --local_files_only \
        --model_dtype bfloat16 \
        --out "$RUN_ROOT/scores/${STEM}__${NAME}.shard0${i}.jsonl" \
        --shard_idx "$i" --num_shards 4 --force >>"$LOG" 2>&1 &
      pids+=($!)
    done
    fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    [[ "$fail" == "0" ]] || { echo "[FATAL] $NAME/$STEM" >&2; tail -60 "$LOG" >&2; exit 1; }
    echo "[ok] $NAME on $STEM"
  done
done

echo "=== summary ==="
RUN_ROOT="$RUN_ROOT" python - <<'PY'
import glob, json, os
import numpy as np
root = os.environ["RUN_ROOT"]
for f in sorted(glob.glob(f"{root}/scores/*.shard00.jsonl")):
    base = f.replace(".shard00.jsonl", "")
    rows = [json.loads(l) for g in glob.glob(base + ".shard*.jsonl") for l in open(g)]
    if not rows:
        continue
    worst = np.array([max(r["scores"]) for r in rows])
    y = np.array([r["correct"] for r in rows])
    if y.any() and not y.all():
        from scipy.stats import rankdata
        rk = rankdata(-worst); n1 = y.sum(); n0 = len(y) - n1
        auroc = (rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    else:
        auroc = float("nan")
    print(f"  {os.path.basename(base)}: n={len(rows)} "
          f"AUROC(worst-step -> correct)={auroc:.3f} "
          f"score med={np.median(np.concatenate([r['scores'] for r in rows])):.4f}")
PY
echo "[$(date)] tts_score done"
