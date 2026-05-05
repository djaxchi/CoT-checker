#!/bin/bash
# Standalone step-6 script: eval + mechanistic analysis + 3-way summary for c=4 TopK SSAE.
# Run from ~/CoT-checker on TamIA after the main training job finishes.
#
# Usage:
#   bash scripts/run_c4_eval.sh
#
# Paths mirror tamia_train_ssae_c4.sh — override via env vars if needed:
#   SCRATCH_RESULTS, C4_EVAL, C4_TRAIN, MECH_DIR

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/CoT-checker}"
STORE="${STORE:-/project/aip-azouaq/$USER}"
SCRATCH_RESULTS="${SCRATCH_RESULTS:-$SCRATCH/cot-checker/results_c4}"
C4_EVAL="${C4_EVAL:-$SCRATCH/cot-checker/probe_data_c4/c4_eval_held_out.npz}"
C4_TRAIN="${C4_TRAIN:-$SCRATCH/cot-checker/probe_data_c4/c4_train_full.npz}"
MECH_DIR="${MECH_DIR:-$PROJECT_DIR/results/mechanistic_c4}"

cd "$PROJECT_DIR"
mkdir -p results

EVAL_OUT="$PROJECT_DIR/results/c4_probe_eval.txt"

# ---------------------------------------------------------------------------
# [1/3] Threshold sweep on all 8 probes (4 linear + 4 MLP)
# ---------------------------------------------------------------------------
echo "=== [1/3] Evaluating c=4 probes on held-out eval set ==="
python - <<PYEOF | tee "$EVAL_OUT"
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import statistics

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
        if not rows:
            continue
        accs  = [r[0]*100 for r in rows]
        f1cs  = [r[1]     for r in rows]
        f1is  = [r[2]     for r in rows]
        macrs = [r[3]     for r in rows]
        def ms(v): return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
        am, as_ = ms(accs);  cm, cs = ms(f1cs);  im, is_ = ms(f1is);  mm, ms_ = ms(macrs)
        print(f"  {t:>5.1f}  {am:>6.2f}±{as_:>4.2f}%  {cm:>6.3f}±{cs:>5.3f}  {im:>6.3f}±{is_:>5.3f}  {mm:>6.3f}±{ms_:>5.3f}")
PYEOF

echo ""

# ---------------------------------------------------------------------------
# [2/3] Mechanistic analysis
# ---------------------------------------------------------------------------
mkdir -p "$MECH_DIR"
echo "=== [2/3] Mechanistic analysis: c=4 TopK sparse h_hat_k ==="
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
# [3/3] Summary: c=4 vs. c=1 sparse vs. dense
# ---------------------------------------------------------------------------
SUMMARY_OUT="$PROJECT_DIR/results/c4_summary.txt"
echo "=== [3/3] Summary: c=4 TopK vs. c=1 sparse vs. Dense h_k ==="
python - <<'PYEOF' | tee "$SUMMARY_OUT"
import glob, re, pathlib, os, statistics

base   = os.path.expandvars("$SCRATCH/cot-checker")
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
    ("c=4 TopK",   r_c4,   "c4_probe_seed*.log",   "c4_linear_probe_seed*.log"),
    ("c=1 sparse", r_main, "probe_seed*.log",        "linear_probe_seed*.log"),
    ("Dense h_k",  r_main, "dense_probe_seed*.log", "dense_linear_probe_seed*.log"),
]

for label, key, lbl2 in [("acc", "Accuracy (%)", "acc"), ("macro_f1", "Macro F1", "macro_f1")]:
    print(f"\n  {lbl2}  --- MLP probe ---")
    print(f"  {'Config':16s}  {'Mean':>12}  {'Std':>8}")
    for name, rdir, mlp_pat, _ in configs:
        rows = parse_logs(rdir, mlp_pat)
        m, s = stats(rows, label)
        if m is None:
            print(f"  {name:16s}  {'N/A':>12}")
        elif label == "acc":
            print(f"  {name:16s}  {m:>11.2f}%  +-{s:>5.2f}%")
        else:
            print(f"  {name:16s}  {m:>12.3f}  +-{s:>5.3f}")

    print(f"\n  {lbl2}  --- Linear probe ---")
    print(f"  {'Config':16s}  {'Mean':>12}  {'Std':>8}")
    for name, rdir, _, lin_pat in configs:
        rows = parse_logs(rdir, lin_pat)
        m, s = stats(rows, label)
        if m is None:
            print(f"  {name:16s}  {'N/A':>12}")
        elif label == "acc":
            print(f"  {name:16s}  {m:>11.2f}%  +-{s:>5.2f}%")
        else:
            print(f"  {name:16s}  {m:>12.3f}  +-{s:>5.3f}")
PYEOF

# ---------------------------------------------------------------------------
# Copy results to persistent project space
# ---------------------------------------------------------------------------
FINAL="/project/aip-azouaq/$USER/results_c4"
CKPT_DIR="/project/aip-azouaq/$USER/checkpoints"
mkdir -p "$FINAL"
cp "$SCRATCH_RESULTS"/c4_probe_seed*.{pt,log}        "$FINAL/" 2>/dev/null || true
cp "$SCRATCH_RESULTS"/c4_linear_probe_seed*.{pt,log} "$FINAL/" 2>/dev/null || true
cp "$SCRATCH/cot-checker/ssae_c4/best.pt" "$CKPT_DIR/ssae_c4_topk40_frozen.pt" 2>/dev/null || true

echo ""
echo "=== Done. Results in $FINAL ==="
echo "  Probe eval saved to  : $EVAL_OUT"
echo "  Summary saved to     : $SUMMARY_OUT"
echo "Fetch all results locally with:"
echo "  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/ ./results/"
