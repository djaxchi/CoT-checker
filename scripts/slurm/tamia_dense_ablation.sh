#!/bin/bash
#SBATCH --job-name=cot-dense-ablation
#SBATCH --account=aip-azouaq
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=h100:4
#SBATCH --output=logs/dense_ablation_%j.out
#SBATCH --error=logs/dense_ablation_%j.err

# ---------------------------------------------------------------------------
# Dense embedding ablation: re-encodes the same Math-Shepherd splits used in
# the main SSAE experiment, but using the raw backbone h_k (no sparse
# bottleneck) to isolate the contribution of the SSAE.
#
# Layout mirrors tamia_generate_data.sh exactly:
#   Eval  : offset   0 →  90K steps  → dense_eval_held_out.npz
#   Train : offset  90K → 450K steps → dense_train_shard_{0..3}.npz → dense_train_full.npz
#
# Then trains 4 probe seeds (same hyperparameters as the main experiment)
# and prints a side-by-side comparison against the SSAE sparse results.
#
# Submit: sbatch scripts/slurm/tamia_dense_ablation.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_DATA="$SCRATCH/cot-checker/probe_data"
SCRATCH_RESULTS="$SCRATCH/cot-checker/results"
CKPT="$STORE/checkpoints/gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"

DENSE_EVAL="$SCRATCH_DATA/dense_eval_held_out.npz"
DENSE_TRAIN="$SCRATCH_DATA/dense_train_full.npz"

cd "$PROJECT_DIR"
mkdir -p logs "$SCRATCH_DATA" "$SCRATCH_RESULTS"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2

source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# [1/4] Eval shard — offset 0, 90K steps (same window as SSAE eval)
# ---------------------------------------------------------------------------
echo "=== [1/4] Encoding eval shard with dense embeddings (GPU 0, offset 0) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/generate_probe_data.py \
    --checkpoint  "$CKPT" \
    --output      "$DENSE_EVAL" \
    --offset      0 \
    --max-steps   90000 \
    --batch-size  64 \
    --max-seq-len 2048 \
    --encoding    dense \
    --device      cuda

echo ""

# ---------------------------------------------------------------------------
# [2/4] Training shards — offset 90K-450K, 4 × 90K (same as SSAE train)
# ---------------------------------------------------------------------------
EVAL_SIZE=90000
SHARD_SIZE=90000
N_SHARDS=4

echo "=== [2/4] Encoding training shards with dense embeddings (4 GPUs in parallel) ==="
PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
    OFFSET=$((EVAL_SIZE + i * SHARD_SIZE))
    OUT="$SCRATCH_DATA/dense_train_shard_${i}.npz"
    echo "  GPU $i: offset=$OFFSET, steps=$SHARD_SIZE → $OUT"
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_probe_data.py \
        --checkpoint  "$CKPT" \
        --output      "$OUT" \
        --offset      "$OFFSET" \
        --max-steps   "$SHARD_SIZE" \
        --batch-size  64 \
        --max-seq-len 2048 \
        --encoding    dense \
        --device      cuda \
        > "$SCRATCH_DATA/dense_shard_${i}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  Dense shard $i finished OK"
    else
        echo "  ERROR: dense shard $i failed — check $SCRATCH_DATA/dense_shard_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED shard(s) failed. Aborting."; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# [3/4] Merge training shards
# ---------------------------------------------------------------------------
echo "=== [3/4] Merging dense training shards ==="
python scripts/slurm/merge_shards.py \
    --inputs  $(for i in $(seq 0 $((N_SHARDS-1))); do echo "$SCRATCH_DATA/dense_train_shard_${i}.npz"; done) \
    --output  "$DENSE_TRAIN"

echo ""

# ---------------------------------------------------------------------------
# [4/5] Linear probe training — 4 seeds in parallel (needed for mechanistic analysis)
# ---------------------------------------------------------------------------
echo "=== [4/5] Training dense linear probes (4 seeds in parallel) ==="
SEEDS=(42 43 44 45)
PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$SCRATCH_RESULTS/dense_linear_probe_seed${SEED}.pt"
    echo "  GPU $i: seed=$SEED → $OUT"
    CUDA_VISIBLE_DEVICES=$i python scripts/experiment_linear_probe.py \
        --train-data "$DENSE_TRAIN" \
        --eval-data  "$DENSE_EVAL" \
        --output     "$OUT" \
        --seed       "$SEED" \
        --epochs     50 \
        --batch-size 512 \
        --device     cuda \
        > "$SCRATCH_RESULTS/dense_linear_probe_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    SEED="${SEEDS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "  Dense linear probe seed $SEED finished OK"
    else
        echo "  ERROR: dense linear probe seed $SEED failed — see $SCRATCH_RESULTS/dense_linear_probe_seed${SEED}.log"
        FAILED=$((FAILED + 1))
    fi
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED linear probe(s) failed."; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# [5/5] MLP probe training — 4 seeds in parallel
# ---------------------------------------------------------------------------
echo "=== [5/5] Training dense MLP probes (4 seeds in parallel) ==="
PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$SCRATCH_RESULTS/dense_probe_seed${SEED}.pt"
    echo "  GPU $i: seed=$SEED → $OUT"
    CUDA_VISIBLE_DEVICES=$i python scripts/experiment_full_clean.py \
        --train-data "$DENSE_TRAIN" \
        --eval-data  "$DENSE_EVAL" \
        --output     "$OUT" \
        --seed       "$SEED" \
        --epochs     50 \
        --batch-size 512 \
        --device     cuda \
        > "$SCRATCH_RESULTS/dense_probe_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    SEED="${SEEDS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "  Dense MLP probe seed $SEED finished OK"
    else
        echo "  ERROR: dense MLP probe seed $SEED failed — see $SCRATCH_RESULTS/dense_probe_seed${SEED}.log"
        FAILED=$((FAILED + 1))
    fi
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED MLP probe(s) failed."; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# Evaluation — run saved probes on the dense held-out set with threshold sweep
# ---------------------------------------------------------------------------
echo "=== Evaluation: dense probes on held-out eval set ==="
python - <<PYEOF
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

results_dir = Path("$SCRATCH_RESULTS")
eval_path   = "$DENSE_EVAL"

# ---- helpers ----
def load_eval(path, seed=42):
    d = np.load(path)
    h, y = d["latents"].astype(np.float32), d["correctness"].astype(np.int64)
    rng = np.random.default_rng(seed)
    cor_idx = np.where(y == 1)[0]
    inc_idx = np.where(y == 0)[0]
    n = min(len(cor_idx), len(inc_idx), 25000)
    sel = np.concatenate([rng.choice(cor_idx, n, replace=False),
                          rng.choice(inc_idx, n, replace=False)])
    rng.shuffle(sel)
    print(f"  Eval: {len(sel):,} steps (balanced 50/50) from {path}")
    return h[sel], y[sel]

def eval_at_threshold(model, h, y, t):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(h)).squeeze(-1)
        pred = (torch.sigmoid(logits) >= t).long().numpy()
    acc = (pred == y).mean()
    def f1(cls):
        tp = ((pred == cls) & (y == cls)).sum()
        fp = ((pred == cls) & (y != cls)).sum()
        fn = ((pred != cls) & (y == cls)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2*p*r/(p+r) if (p+r) else 0.0
    return acc, f1(1), f1(0)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
        )
    def forward(self, x): return self.net(x)

class Linear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x): return self.fc(x)

def load_mlp(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    m = MLP(cfg["input_dim"], cfg["hidden_dim"])
    m.load_state_dict(ckpt["model"])
    return m

def load_linear(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd   = ckpt["state_dict"]
    m = Linear(sd["fc.weight"].shape[1])
    m.load_state_dict(sd)
    return m

h_eval, y_eval = load_eval(eval_path)
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
seeds = [42, 43, 44, 45]

for probe_type, loader, prefix in [
    ("MLP",    load_mlp,    "dense_probe_seed"),
    ("Linear", load_linear, "dense_linear_probe_seed"),
]:
    print(f"\n{'='*62}")
    print(f"  Dense {probe_type} probe — threshold sweep (4 seeds)")
    print(f"{'='*62}")
    print(f"  {'Seed':>4}  {'t':>5}  {'Acc':>7}  {'F1 cor':>8}  {'F1 inc':>8}  {'Macro':>8}")
    print(f"  {'─'*58}")

    per_threshold = {t: [] for t in thresholds}
    for seed in seeds:
        pt = results_dir / f"{prefix}{seed}.pt"
        if not pt.exists():
            print(f"  WARNING: {pt} not found, skipping seed {seed}")
            continue
        model = loader(str(pt))
        for t in thresholds:
            acc, f1c, f1i = eval_at_threshold(model, h_eval, y_eval, t)
            macro = (f1c + f1i) / 2
            per_threshold[t].append((acc, f1c, f1i, macro))
            print(f"  {seed:>4}  {t:>5.1f}  {acc*100:>6.2f}%  {f1c:>8.3f}  {f1i:>8.3f}  {macro:>8.3f}")

    print(f"\n  {'─'*58}")
    print(f"  Mean ± std across seeds:")
    print(f"  {'t':>5}  {'Acc':>12}  {'F1 cor':>12}  {'F1 inc':>12}  {'Macro':>12}")
    for t in thresholds:
        rows = per_threshold[t]
        if not rows: continue
        import statistics
        accs  = [r[0]*100 for r in rows]
        f1cs  = [r[1]     for r in rows]
        f1is  = [r[2]     for r in rows]
        macrs = [r[3]     for r in rows]
        def ms(v): return statistics.mean(v), (statistics.stdev(v) if len(v)>1 else 0.0)
        am, as_ = ms(accs);  cm, cs = ms(f1cs);  im, is_ = ms(f1is);  mm, ms_ = ms(macrs)
        print(f"  {t:>5.1f}  {am:>6.2f}±{as_:>4.2f}%  {cm:>6.3f}±{cs:>5.3f}  {im:>6.3f}±{is_:>5.3f}  {mm:>6.3f}±{ms_:>5.3f}")
PYEOF

echo ""

# ---------------------------------------------------------------------------
# Mechanistic analysis on dense embeddings (CPU-only, runs after probes finish)
# ---------------------------------------------------------------------------
DENSE_MECH_DIR="$PROJECT_DIR/results/mechanistic_dense"
mkdir -p "$DENSE_MECH_DIR"

echo "=== Mechanistic analysis: Dense h_k ==="
python scripts/mechanistic_analysis.py \
    --eval-data  "$DENSE_EVAL" \
    --train-data "$DENSE_TRAIN" \
    --probes     "$SCRATCH_RESULTS/dense_linear_probe_seed42.pt" \
                 "$SCRATCH_RESULTS/dense_linear_probe_seed43.pt" \
                 "$SCRATCH_RESULTS/dense_linear_probe_seed44.pt" \
                 "$SCRATCH_RESULTS/dense_linear_probe_seed45.pt" \
    --output-dir "$DENSE_MECH_DIR" \
    --label      "Dense h_k" \
    --max-train-samples 50000

echo ""
echo "  Dense mechanistic plots saved to $DENSE_MECH_DIR"

# ---------------------------------------------------------------------------
# Summary: dense vs. SSAE sparse side-by-side
# ---------------------------------------------------------------------------
echo ""
echo "=== Summary: SSAE sparse vs. dense embedding ==="
python - <<'PYEOF'
import glob, re, pathlib, os, statistics

results_dir = os.path.expandvars("$SCRATCH/cot-checker/results")

def parse_logs(pattern):
    logs = sorted(glob.glob(f"{results_dir}/{pattern}"))
    rows = []
    for log in logs:
        text = pathlib.Path(log).read_text()
        m = re.search(r"^SUMMARY (.+)$", text, re.MULTILINE)
        if not m:
            continue
        kv = dict(item.split("=") for item in m.group(1).split())
        rows.append(kv)
    return rows

def stats(rows, key):
    vals = [float(r[key]) for r in rows]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

print("\n=== MLP probe: SSAE sparse vs. Dense h_k ===")
sparse_rows = parse_logs("probe_seed*.log")
dense_rows  = parse_logs("dense_probe_seed*.log")

print(f"\n{'':36s}  {'SSAE sparse':>14}  {'Dense h_k':>14}")
print("─" * 68)
for key, lbl in [
    ("acc",           "Accuracy (%)"),
    ("f1_correct",    "F1 correct"),
    ("f1_incorrect",  "F1 incorrect"),
    ("macro_f1",      "Macro F1"),
]:
    s_str = "N/A"
    d_str = "N/A"
    if sparse_rows:
        sm, ss = stats(sparse_rows, key)
        s_str = f"{sm:.2f} ± {ss:.2f}" if key == "acc" else f"{sm:.3f} ± {ss:.3f}"
    if dense_rows:
        dm, ds = stats(dense_rows, key)
        d_str = f"{dm:.2f} ± {ds:.2f}" if key == "acc" else f"{dm:.3f} ± {ds:.3f}"
    print(f"  {lbl:34s}  {s_str:>14}  {d_str:>14}")

if sparse_rows and dense_rows:
    sa, _ = stats(sparse_rows, "acc")
    da, _ = stats(dense_rows,  "acc")
    print(f"\n  SSAE advantage (MLP): {(sa - da):+.2f}pp accuracy")

print("\n=== Linear probe: SSAE sparse vs. Dense h_k ===")
sparse_lin = parse_logs("linear_probe_seed*.log")
dense_lin  = parse_logs("dense_linear_probe_seed*.log")

print(f"\n{'':36s}  {'SSAE sparse':>14}  {'Dense h_k':>14}")
print("─" * 68)
for key, lbl in [
    ("acc",           "Accuracy (%)"),
    ("macro_f1",      "Macro F1"),
]:
    s_str = "N/A"
    d_str = "N/A"
    if sparse_lin:
        sm, ss = stats(sparse_lin, key)
        s_str = f"{sm:.2f} ± {ss:.2f}" if key == "acc" else f"{sm:.3f} ± {ss:.3f}"
    if dense_lin:
        dm, ds = stats(dense_lin, key)
        d_str = f"{dm:.2f} ± {ds:.2f}" if key == "acc" else f"{dm:.3f} ± {ds:.3f}"
    print(f"  {lbl:34s}  {s_str:>14}  {d_str:>14}")

if sparse_lin and dense_lin:
    sa, _ = stats(sparse_lin, "acc")
    da, _ = stats(dense_lin,  "acc")
    print(f"\n  SSAE advantage (linear): {(sa - da):+.2f}pp accuracy")
PYEOF

# ---------------------------------------------------------------------------
# Copy all results to project space
# ---------------------------------------------------------------------------
FINAL="$STORE/results"
mkdir -p "$FINAL"
cp "$SCRATCH_RESULTS"/dense_probe_seed*.{pt,log}        "$FINAL/" 2>/dev/null || true
cp "$SCRATCH_RESULTS"/dense_linear_probe_seed*.{pt,log} "$FINAL/" 2>/dev/null || true
echo ""
echo "=== Done. Results in $FINAL ==="
echo "Fetch dense mechanistic plots with:"
echo "  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_dense/ ./results/mechanistic_dense/"
