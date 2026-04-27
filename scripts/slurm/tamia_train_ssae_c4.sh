#!/bin/bash
#SBATCH --job-name=cot-ssae-c4
#SBATCH --account=aip-azouaq
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=h100:4
#SBATCH --output=logs/ssae_c4_%j.out
#SBATCH --error=logs/ssae_c4_%j.err

# ---------------------------------------------------------------------------
# Full pipeline for overcomplete SSAE (c=4, TopK-K=40, frozen encoder).
#
# Hypothesis: sparsity_factor=4 gives an overcomplete 3584-dim dictionary
# where correctness-relevant features are disentangled, so probes trained on
# h_hat_k should outperform the dense baseline and the c=1 baseline.
#
# Steps:
#   [1/6] Train SSAE phase 1: c=4, TopK-40, frozen encoder (GPU 0, ~15-20h)
#   [2/6] Encode eval shard    (offset 0,    90K steps, GPU 0)
#   [3/6] Encode train shards  (offset 90K-450K, 4 × 90K, 4 GPUs parallel)
#   [4/6] Merge training shards
#   [5/6] Train linear + MLP probes (4 seeds each, 4 GPUs parallel)
#   [6/6] Evaluate, mechanistic analysis, summary vs. c=1 and dense baselines
#
# Prerequisites (run once on login node before submitting):
#   Data must be at $HOME/CoT-checker/data/gsm8k_385K_train.json
#   and $HOME/CoT-checker/data/gsm8k_385K_valid.json
#   These are gitignored (148 MB) — copy them to TamIA manually if absent.
#
# Submit: sbatch scripts/slurm/tamia_train_ssae_c4.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_DATA="$SCRATCH/cot-checker/probe_data_c4"
SCRATCH_RESULTS="$SCRATCH/cot-checker/results_c4"
CKPT_DIR="$STORE/checkpoints"
NEW_CKPT_DIR="$SCRATCH/cot-checker/ssae_c4"

TRAIN_DATA="$STORE/data/gsm8k_385K_train.json"
VAL_DATA="$STORE/data/gsm8k_385K_valid.json"
NEW_CKPT="$NEW_CKPT_DIR/best.pt"

C4_EVAL="$SCRATCH_DATA/c4_eval_held_out.npz"
C4_TRAIN="$SCRATCH_DATA/c4_train_full.npz"

cd "$PROJECT_DIR"
mkdir -p logs "$SCRATCH_DATA" "$SCRATCH_RESULTS" "$NEW_CKPT_DIR"

module purge
module load StdEnv/2023 gcc arrow/24.0.0 python/3.11 cuda/12.2

source "$HOME/venvs/cot/bin/activate"

export HF_HOME="$STORE/hf_cache"
export TRANSFORMERS_CACHE="$STORE/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: $TRAIN_DATA not found. Aborting."
    exit 1
fi
if [ ! -f "$VAL_DATA" ]; then
    echo "WARNING: $VAL_DATA not found — will auto-split 5% of train for validation."
    VAL_FLAG=""
else
    VAL_FLAG="--val-data $VAL_DATA"
fi
echo "Training data found: $TRAIN_DATA"

# ---------------------------------------------------------------------------
# [1/6] Train SSAE — c=4, TopK-K=40, frozen encoder, bfloat16, 10 epochs
#
# Trainable: autoencoder (W: 896x3584), projection_mlp (3584→3584→3584), decoder
# Frozen:    encoder backbone (~125M params → ~2x faster backward)
# L1 penalty: disabled (TopK enforces exact-K sparsity structurally)
# ---------------------------------------------------------------------------
echo "=== [1/6] Training SSAE c=4, TopK-40, frozen encoder (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/train_ssae.py \
    --data          "$TRAIN_DATA" \
    ${VAL_FLAG:+--val-data "$VAL_DATA"} \
    --output-dir    "$NEW_CKPT_DIR" \
    --model-id      Qwen/Qwen2.5-0.5B \
    --sparsity-factor 4 \
    --topk-k        40 \
    --freeze-encoder \
    --dtype         bfloat16 \
    --epochs        10 \
    --batch-size    32 \
    --grad-accum    4 \
    --lr            1e-4 \
    --min-lr        1e-5 \
    --warmup-steps  500 \
    --num-workers   4 \
    --device        cuda

echo "SSAE training complete. Checkpoint at $NEW_CKPT"
echo ""

# ---------------------------------------------------------------------------
# [2/6] Encode eval shard with the new c=4 checkpoint
# ---------------------------------------------------------------------------
echo "=== [2/6] Encoding eval shard (GPU 0, offset 0, 90K steps) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/generate_probe_data.py \
    --checkpoint  "$NEW_CKPT" \
    --output      "$C4_EVAL" \
    --offset      0 \
    --max-steps   90000 \
    --batch-size  32 \
    --max-seq-len 2048 \
    --encoding    sparse \
    --device      cuda

echo ""

# ---------------------------------------------------------------------------
# [3/6] Encode training shards (4 x 90K, 4 GPUs in parallel)
# ---------------------------------------------------------------------------
EVAL_SIZE=90000
SHARD_SIZE=90000
N_SHARDS=4

echo "=== [3/6] Encoding training shards (4 GPUs in parallel) ==="
PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
    OFFSET=$((EVAL_SIZE + i * SHARD_SIZE))
    OUT="$SCRATCH_DATA/c4_train_shard_${i}.npz"
    echo "  GPU $i: offset=$OFFSET, steps=$SHARD_SIZE → $OUT"
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_probe_data.py \
        --checkpoint  "$NEW_CKPT" \
        --output      "$OUT" \
        --offset      "$OFFSET" \
        --max-steps   "$SHARD_SIZE" \
        --batch-size  32 \
        --max-seq-len 2048 \
        --encoding    sparse \
        --device      cuda \
        > "$SCRATCH_DATA/c4_shard_${i}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  Shard $i finished OK"
    else
        echo "  ERROR: shard $i failed — check $SCRATCH_DATA/c4_shard_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED shard(s) failed. Aborting."; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# [4/6] Merge training shards
# ---------------------------------------------------------------------------
echo "=== [4/6] Merging c=4 training shards ==="
python scripts/slurm/merge_shards.py \
    --inputs  $(for i in $(seq 0 $((N_SHARDS-1))); do echo "$SCRATCH_DATA/c4_train_shard_${i}.npz"; done) \
    --output  "$C4_TRAIN"

echo ""

# ---------------------------------------------------------------------------
# [5/6] Train probes — linear and MLP, 4 seeds in parallel
# ---------------------------------------------------------------------------
SEEDS=(42 43 44 45)

echo "=== [5/6a] Training c=4 linear probes (4 seeds in parallel) ==="
PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$SCRATCH_RESULTS/c4_linear_probe_seed${SEED}.pt"
    CUDA_VISIBLE_DEVICES=$i python scripts/experiment_linear_probe.py \
        --train-data "$C4_TRAIN" \
        --eval-data  "$C4_EVAL" \
        --output     "$OUT" \
        --seed       "$SEED" \
        --epochs     50 \
        --batch-size 512 \
        --device     cuda \
        > "$SCRATCH_RESULTS/c4_linear_probe_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    SEED="${SEEDS[$i]}"
    wait "${PIDS[$i]}" && echo "  Linear probe seed $SEED OK" \
        || { echo "  ERROR: linear probe seed $SEED failed"; FAILED=$((FAILED + 1)); }
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED linear probe(s) failed."; exit 1; }

echo ""
echo "=== [5/6b] Training c=4 MLP probes (4 seeds in parallel) ==="
PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$SCRATCH_RESULTS/c4_probe_seed${SEED}.pt"
    CUDA_VISIBLE_DEVICES=$i python scripts/experiment_full_clean.py \
        --train-data "$C4_TRAIN" \
        --eval-data  "$C4_EVAL" \
        --output     "$OUT" \
        --seed       "$SEED" \
        --epochs     50 \
        --batch-size 512 \
        --device     cuda \
        > "$SCRATCH_RESULTS/c4_probe_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    SEED="${SEEDS[$i]}"
    wait "${PIDS[$i]}" && echo "  MLP probe seed $SEED OK" \
        || { echo "  ERROR: MLP probe seed $SEED failed"; FAILED=$((FAILED + 1)); }
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED MLP probe(s) failed."; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# [6/6] Evaluation — threshold sweep on all 8 probes
# ---------------------------------------------------------------------------
echo "=== [6/6] Evaluating c=4 probes on held-out eval set ==="
python - <<PYEOF
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

results_dir = Path("$SCRATCH_RESULTS")
eval_path   = "$C4_EVAL"

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
    ("MLP",    load_mlp,    "c4_probe_seed"),
    ("Linear", load_linear, "c4_linear_probe_seed"),
]:
    print(f"\n{'='*62}")
    print(f"  c=4 TopK {probe_type} probe — threshold sweep (4 seeds)")
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
# Mechanistic analysis on c=4 sparse vectors
# ---------------------------------------------------------------------------
MECH_DIR="$PROJECT_DIR/results/mechanistic_c4"
mkdir -p "$MECH_DIR"

echo "=== Mechanistic analysis: c=4 TopK sparse h_hat_k ==="
python scripts/mechanistic_analysis.py \
    --eval-data  "$C4_EVAL" \
    --train-data "$C4_TRAIN" \
    --probes     "$SCRATCH_RESULTS/c4_linear_probe_seed42.pt" \
                 "$SCRATCH_RESULTS/c4_linear_probe_seed43.pt" \
                 "$SCRATCH_RESULTS/c4_linear_probe_seed44.pt" \
                 "$SCRATCH_RESULTS/c4_linear_probe_seed45.pt" \
    --output-dir "$MECH_DIR" \
    --label      "SSAE c=4 TopK-40" \
    --max-train-samples 50000

echo "  c=4 mechanistic plots saved to $MECH_DIR"
echo ""

# ---------------------------------------------------------------------------
# Summary: c=4 vs. c=1 sparse vs. dense
# ---------------------------------------------------------------------------
echo "=== Summary: c=4 TopK vs. c=1 sparse vs. Dense h_k ==="
python - <<'PYEOF'
import glob, re, pathlib, os, statistics

base = os.path.expandvars("$SCRATCH/cot-checker")
r_main = os.path.join(base, "results")
r_c4   = os.path.join(base, "results_c4")

def parse_logs(directory, pattern):
    logs = sorted(glob.glob(os.path.join(directory, pattern)))
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
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

configs = [
    ("c=4 TopK",   r_c4,   "c4_probe_seed*.log",      "c4_linear_probe_seed*.log"),
    ("c=1 sparse", r_main, "probe_seed*.log",           "linear_probe_seed*.log"),
    ("Dense h_k",  r_main, "dense_probe_seed*.log",    "dense_linear_probe_seed*.log"),
]

for label, key, lbl2 in [("acc", "Accuracy (%)", "acc"), ("macro_f1", "Macro F1", "macro_f1")]:
    print(f"\n  {lbl2}  ─── MLP probe ───────────────────────────────────────")
    print(f"  {'Config':16s}  {'Mean':>12}  {'Std':>8}")
    for name, rdir, mlp_pat, _ in configs:
        rows = parse_logs(rdir, mlp_pat)
        m, s = stats(rows, label)
        if m is None:
            print(f"  {name:16s}  {'N/A':>12}")
        elif label == "acc":
            print(f"  {name:16s}  {m:>11.2f}%  ±{s:>6.2f}%")
        else:
            print(f"  {name:16s}  {m:>12.3f}  ±{s:>6.3f}")

    print(f"\n  {lbl2}  ─── Linear probe ────────────────────────────────────")
    print(f"  {'Config':16s}  {'Mean':>12}  {'Std':>8}")
    for name, rdir, _, lin_pat in configs:
        rows = parse_logs(rdir, lin_pat)
        m, s = stats(rows, label)
        if m is None:
            print(f"  {name:16s}  {'N/A':>12}")
        elif label == "acc":
            print(f"  {name:16s}  {m:>11.2f}%  ±{s:>6.2f}%")
        else:
            print(f"  {name:16s}  {m:>12.3f}  ±{s:>6.3f}")
PYEOF

# ---------------------------------------------------------------------------
# Copy results to project space
# ---------------------------------------------------------------------------
FINAL="$STORE/results_c4"
mkdir -p "$FINAL"
cp "$SCRATCH_RESULTS"/c4_probe_seed*.{pt,log}        "$FINAL/" 2>/dev/null || true
cp "$SCRATCH_RESULTS"/c4_linear_probe_seed*.{pt,log} "$FINAL/" 2>/dev/null || true
cp "$NEW_CKPT" "$CKPT_DIR/ssae_c4_topk40_frozen.pt"  2>/dev/null || true
echo ""
echo "=== Done. Results in $FINAL ==="
echo "Fetch mechanistic plots with:"
echo "  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_c4/ ./results/mechanistic_c4/"
echo "Checkpoint saved to $CKPT_DIR/ssae_c4_topk40_frozen.pt"
