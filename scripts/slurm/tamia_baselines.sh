#!/bin/bash
#SBATCH --job-name=cot-baselines
#SBATCH --account=aip-azouaq
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=h100:4
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

# ---------------------------------------------------------------------------
# Baselines: (1) linear probe across 4 seeds; (2) LLM self-judge on 50K eval
# for Qwen2.5-0.5B-Instruct and Qwen2.5-Math-7B-Instruct.
#
# Phase 1:  4 parallel linear probe seeds on GPUs 0-3 (~2 min).
# Phase 2a: 4 parallel shards of Qwen-0.5B on the 50K eval (GPUs 0-3).
# Phase 2b: 4 parallel shards of Qwen-7B on the 50K eval (GPUs 0-3).
# Phase 3:  Aggregate everything into a summary.
#
# Requires: Qwen checkpoints already in $STORE/hf_cache (run
# scripts/slurm/tamia_download_qwen.sh on the login node first).
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="$HOME/projects/aip-azouaq/$USER"
SCRATCH_RESULTS="$SCRATCH/cot-checker/results"
HF_CACHE="$STORE/hf_cache"
RECORDS_CACHE="$SCRATCH/cot-checker/probe_data/eval_held_out_text.jsonl"

TRAIN_DATA="$STORE/probe_data/train_final.npz"
EVAL_DATA="$STORE/probe_data/eval_held_out.npz"

cd "$PROJECT_DIR"
mkdir -p logs "$SCRATCH_RESULTS"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2

source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# Phase 1: linear probe (4 seeds in parallel)
# ---------------------------------------------------------------------------
echo "=== [Phase 1] Linear probe: 4 seeds in parallel ==="
SEEDS=(42 43 44 45)
PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$SCRATCH_RESULTS/linear_probe_seed${SEED}.pt"
    echo "  GPU $i: linear seed=$SEED -> $OUT"
    CUDA_VISIBLE_DEVICES=$i python scripts/experiment_linear_probe.py \
        --train-data "$TRAIN_DATA" \
        --eval-data  "$EVAL_DATA" \
        --output     "$OUT" \
        --seed       "$SEED" \
        --epochs     50 \
        --batch-size 512 \
        --device cuda \
        > "$SCRATCH_RESULTS/linear_probe_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    SEED="${SEEDS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "  Linear seed $SEED finished OK"
    else
        echo "  ERROR: linear seed $SEED failed -- see $SCRATCH_RESULTS/linear_probe_seed${SEED}.log"
        FAILED=$((FAILED + 1))
    fi
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED linear probe(s) failed"; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# Pre-build the shared eval text records once so all LLM shards skip re-parsing.
# Needs internet: if this fails, you must run it on the login node beforehand.
# ---------------------------------------------------------------------------
if [ ! -f "$RECORDS_CACHE" ]; then
    echo "=== Building eval text records (one-time) ==="
    # Temporarily allow online for dataset streaming if it's the first time.
    # (If compute node has no internet and this fails, pre-run on login node.)
    HF_DATASETS_OFFLINE=0 python -c "
import sys; sys.path.insert(0, '.')
from scripts.llm_self_judge import collect_eval_text_records
import json, os
recs, _ = collect_eval_text_records()
os.makedirs(os.path.dirname('$RECORDS_CACHE'), exist_ok=True)
with open('$RECORDS_CACHE', 'w') as f:
    for r in recs: f.write(json.dumps(r) + '\n')
print('Cached', len(recs), 'records ->', '$RECORDS_CACHE')
"
fi

# ---------------------------------------------------------------------------
# Phase 2a: Qwen2.5-0.5B-Instruct, 4 shards in parallel
# ---------------------------------------------------------------------------
MODEL_SMALL="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_BIG="Qwen/Qwen2.5-Math-7B-Instruct"

echo "=== [Phase 2a] LLM self-judge: $MODEL_SMALL (4 shards) ==="
PIDS=()
for i in 0 1 2 3; do
    OUT="$SCRATCH_RESULTS/judge_qwen05b_shard${i}.npz"
    echo "  GPU $i: shard $i/4 -> $OUT"
    CUDA_VISIBLE_DEVICES=$i python scripts/llm_self_judge.py \
        --model "$MODEL_SMALL" \
        --output "$OUT" \
        --shard-idx $i --n-shards 4 \
        --batch-size 32 \
        --dtype bfloat16 \
        --cache-dir "$HF_CACHE" \
        --records-cache "$RECORDS_CACHE" \
        --device cuda \
        > "$SCRATCH_RESULTS/judge_qwen05b_shard${i}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  0.5B shard $i finished OK"
    else
        echo "  ERROR: 0.5B shard $i failed -- see $SCRATCH_RESULTS/judge_qwen05b_shard${i}.log"
        FAILED=$((FAILED + 1))
    fi
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED Qwen-0.5B shard(s) failed"; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# Phase 2b: Qwen2.5-Math-7B-Instruct, 4 shards in parallel
# Full precision (bfloat16). 7B easily fits on a single 80GB H100.
# ---------------------------------------------------------------------------
echo "=== [Phase 2b] LLM self-judge: $MODEL_BIG (4 shards) ==="
PIDS=()
for i in 0 1 2 3; do
    OUT="$SCRATCH_RESULTS/judge_qwen7b_shard${i}.npz"
    echo "  GPU $i: shard $i/4 -> $OUT"
    CUDA_VISIBLE_DEVICES=$i python scripts/llm_self_judge.py \
        --model "$MODEL_BIG" \
        --output "$OUT" \
        --shard-idx $i --n-shards 4 \
        --batch-size 8 \
        --dtype bfloat16 \
        --cache-dir "$HF_CACHE" \
        --records-cache "$RECORDS_CACHE" \
        --device cuda \
        > "$SCRATCH_RESULTS/judge_qwen7b_shard${i}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  7B shard $i finished OK"
    else
        echo "  ERROR: 7B shard $i failed -- see $SCRATCH_RESULTS/judge_qwen7b_shard${i}.log"
        FAILED=$((FAILED + 1))
    fi
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED Qwen-7B shard(s) failed"; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# Phase 3: aggregate everything
# ---------------------------------------------------------------------------
echo "=== [Phase 3] Aggregating results ==="
python - <<PYEOF
import glob, re, pathlib, os, statistics
import numpy as np

results_dir = "$SCRATCH_RESULTS"

# ---- Linear probes ----
print("\n--- Linear probe (4 seeds) ---")
logs = sorted(glob.glob(f"{results_dir}/linear_probe_seed*.log"))
rows = []
for log in logs:
    text = pathlib.Path(log).read_text()
    m = re.search(r"^SUMMARY (.+)$", text, re.MULTILINE)
    if not m: continue
    kv = dict(item.split("=") for item in m.group(1).split())
    rows.append(kv)

def stats(rows, key):
    vals = [float(r[key]) for r in rows]
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)

if rows:
    for r in rows:
        print(f"  seed={r['seed']:>2}  acc={float(r['acc']):.2f}%  macro_f1={float(r['macro_f1']):.3f}  "
              f"f1_correct={float(r['f1_correct']):.3f}  f1_incorrect={float(r['f1_incorrect']):.3f}")
    am, as_ = stats(rows, "acc")
    mm, ms = stats(rows, "macro_f1")
    print(f"\n  Accuracy  : {am:.2f}% +/- {as_:.2f}")
    print(f"  Macro F1  : {mm:.3f} +/- {ms:.3f}")

# ---- LLM self-judge ----
def eval_judge(name, pattern):
    files = sorted(glob.glob(f"{results_dir}/{pattern}"))
    if not files:
        print(f"\n  No shards for {name}"); return
    all_p, all_y = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        all_p.append(d["p_yes"]); all_y.append(d["labels"])
    p = np.concatenate(all_p); y = np.concatenate(all_y).astype(np.int64)
    print(f"\n--- {name} ({len(y):,} steps) ---")
    print(f"  {'Threshold':>10}  {'Accuracy':>10}  {'F1 correct':>12}  {'F1 incorrect':>13}  {'Macro F1':>10}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (p >= t).astype(np.int64)
        acc = (pred == y).mean()
        def f1(cls):
            tp = ((pred == cls) & (y == cls)).sum()
            fp = ((pred == cls) & (y != cls)).sum()
            fn = ((pred != cls) & (y == cls)).sum()
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            return 2*prec*rec/(prec+rec) if (prec+rec) else 0
        fc, fi = f1(1), f1(0)
        print(f"  {t:>10.1f}  {acc*100:>9.2f}%  {fc:>12.3f}  {fi:>13.3f}  {(fc+fi)/2:>10.3f}")

eval_judge("Qwen2.5-0.5B-Instruct",  "judge_qwen05b_shard*.npz")
eval_judge("Qwen2.5-Math-7B-Instruct", "judge_qwen7b_shard*.npz")
PYEOF

# Copy to project space
FINAL="$STORE/results"
mkdir -p "$FINAL"
cp "$SCRATCH_RESULTS"/linear_probe_seed*.{pt,log}  "$FINAL/" 2>/dev/null || true
cp "$SCRATCH_RESULTS"/judge_qwen*.{npz,log}        "$FINAL/" 2>/dev/null || true
echo ""
echo "=== Done. Results in $FINAL ==="
ls -lh "$FINAL" | head -40
