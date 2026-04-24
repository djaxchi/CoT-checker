#!/bin/bash
#SBATCH --job-name=cot-probe
#SBATCH --account=aip-azouaq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # 4 runs × 4 workers each
#SBATCH --mem=64G
#SBATCH --gpus=h100:4               # must use all 4 GPUs on an H100 node
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err

# ---------------------------------------------------------------------------
# Probe training: runs 4 seeds in parallel (one per GPU) to get mean ± std.
# This satisfies TamIA's requirement to use all allocated GPUs.
#
# Run AFTER tamia_generate_data.sh has completed.
#
# Submit: sbatch scripts/slurm/tamia_train_probe.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_RESULTS="$SCRATCH/cot-checker/results"

TRAIN_DATA="$STORE/probe_data/train_full.npz"
EVAL_DATA="$STORE/probe_data/eval_held_out.npz"

cd "$PROJECT_DIR"
mkdir -p logs "$SCRATCH_RESULTS"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2

source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== Training 4 probe seeds in parallel (one per GPU) ==="
SEEDS=(42 43 44 45)
PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$SCRATCH_RESULTS/probe_seed${SEED}.pt"
    echo "  GPU $i: seed=$SEED → $OUT"
    CUDA_VISIBLE_DEVICES=$i python scripts/experiment_full_clean.py \
        --train-data "$TRAIN_DATA" \
        --eval-data  "$EVAL_DATA" \
        --output     "$OUT" \
        --seed       "$SEED" \
        --epochs     50 \
        --batch-size 512 \
        --device cuda \
        > "$SCRATCH_RESULTS/probe_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    SEED="${SEEDS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "  Seed $SEED finished OK"
    else
        echo "  ERROR: seed $SEED failed — check $SCRATCH_RESULTS/probe_seed${SEED}.log"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED run(s) failed."
    exit 1
fi

echo ""
echo "=== Aggregating results across seeds ==="
python - <<'PYEOF'
import glob, re, pathlib, os, statistics

results_dir = os.path.expandvars("$SCRATCH/cot-checker/results")
logs = sorted(glob.glob(f"{results_dir}/probe_seed*.log"))

rows = []
for log in logs:
    text = pathlib.Path(log).read_text()
    # Each run emits a single machine-readable SUMMARY line, e.g.:
    # SUMMARY seed=42 acc=74.31 gain=24.31 f1_correct=0.712 f1_incorrect=0.756 ...
    m = re.search(r"^SUMMARY (.+)$", text, re.MULTILINE)
    if not m:
        print(f"  WARNING: no SUMMARY line in {pathlib.Path(log).name} — run may have failed")
        continue
    kv = dict(item.split("=") for item in m.group(1).split())
    rows.append(kv)
    print(
        f"  seed={kv['seed']:>2}"
        f"  acc={float(kv['acc']):.2f}%"
        f"  gain=+{float(kv['gain']):.2f}pp"
        f"  F1_correct={float(kv['f1_correct']):.3f}"
        f"  F1_incorrect={float(kv['f1_incorrect']):.3f}"
        f"  macro_F1={float(kv['macro_f1']):.3f}"
        f"  train={float(kv['train_sec']):.0f}s"
        f"  latency_mean={float(kv['latency_mean_ms']):.3f}ms"
        f"  latency_p99={float(kv['latency_p99_ms']):.3f}ms"
        f"  throughput={float(kv['throughput_per_sec']):.0f} steps/s"
    )

if not rows:
    print("No results parsed — check individual log files.")
else:
    def stats(key):
        vals = [float(r[key]) for r in rows]
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    print(f"\n{'─'*60}")
    print(f"  Aggregated over {len(rows)} seed(s)")
    print(f"{'─'*60}")
    acc_m,  acc_s  = stats("acc")
    gain_m, gain_s = stats("gain")
    f1c_m,  f1c_s  = stats("f1_correct")
    f1i_m,  f1i_s  = stats("f1_incorrect")
    mf1_m,  mf1_s  = stats("macro_f1")
    tr_m,   tr_s   = stats("train_sec")
    lat_m,  lat_s  = stats("latency_mean_ms")
    p99_m,  p99_s  = stats("latency_p99_ms")
    thr_m,  thr_s  = stats("throughput_per_sec")

    print(f"  Accuracy         : {acc_m:.2f}% ± {acc_s:.2f}%")
    print(f"  Gain vs majority : +{gain_m:.2f}pp ± {gain_s:.2f}pp")
    print(f"  F1 (correct)     : {f1c_m:.3f} ± {f1c_s:.3f}")
    print(f"  F1 (incorrect)   : {f1i_m:.3f} ± {f1i_s:.3f}")
    print(f"  Macro F1         : {mf1_m:.3f} ± {mf1_s:.3f}")
    print(f"\n  Training time    : {tr_m:.0f}s ± {tr_s:.0f}s")
    print(f"  Latency (mean)   : {lat_m:.3f}ms ± {lat_s:.3f}ms")
    print(f"  Latency (p99)    : {p99_m:.3f}ms ± {p99_s:.3f}ms")
    print(f"  Throughput       : {thr_m:.0f} ± {thr_s:.0f} steps/sec")
PYEOF

# Copy results to project space
FINAL="$STORE/results"
mkdir -p "$FINAL"
cp "$SCRATCH_RESULTS"/probe_seed*.pt  "$FINAL/" 2>/dev/null || true
cp "$SCRATCH_RESULTS"/probe_seed*.log "$FINAL/" 2>/dev/null || true
echo ""
echo "Results saved to $FINAL"
ls -lh "$FINAL"
