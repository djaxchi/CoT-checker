#!/bin/bash
#SBATCH --job-name=cot-ssae-c2
#SBATCH --account=aip-azouaq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=h100:4
#SBATCH --output=logs/ssae_c2_%j.out
#SBATCH --error=logs/ssae_c2_%j.err

# ---------------------------------------------------------------------------
# Full pipeline for overcomplete SSAE c=2 with ReLU + L1 sparsity.
#
# Key difference from c=4: no TopK, no frozen encoder.
# sparsity_factor=2 gives a 1792-dim dictionary (2x overcomplete).
# ReLU + L1 penalty lets the model use however many features each step
# actually needs, instead of forcing a fixed k -- this is the right
# inductive bias for monosemanticity.
#
# Steps:
#   [1/6] Train SSAE phase 1: c=2, ReLU+L1, full encoder (GPU 0)
#   [2/6] Encode eval shard   (90K steps, GPU 0)
#   [3/6] Encode train shards (4 × 90K, 4 GPUs parallel)
#   [4/6] Merge training shards
#   [5/6] Train linear + MLP probes (4 seeds, 4 GPUs parallel)
#   [6/6] Eval + mechanistic analysis + print summary
#
# Submit: sbatch scripts/slurm/tamia_train_ssae_c2.sh
# After: sbatch scripts/slurm/tamia_c2_encode_and_compare.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="$HOME/CoT-checker"
STORE="/project/aip-azouaq/$USER"
SCRATCH_DATA="$SCRATCH/cot-checker/probe_data_c2"
SCRATCH_RESULTS="$SCRATCH/cot-checker/results_c2"
CKPT_DIR="$STORE/checkpoints"
NEW_CKPT_DIR="$SCRATCH/cot-checker/ssae_c2"

TRAIN_DATA="$STORE/data/gsm8k_385K_train.json"
VAL_DATA="$STORE/data/gsm8k_385K_valid.json"
NEW_CKPT="$NEW_CKPT_DIR/best.pt"

C2_EVAL="$SCRATCH_DATA/c2_eval_held_out.npz"
C2_TRAIN="$SCRATCH_DATA/c2_train_full.npz"

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
# Pre-flight
# ---------------------------------------------------------------------------
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: $TRAIN_DATA not found. Aborting."
    exit 1
fi
if [ -f "$VAL_DATA" ]; then
    VAL_FLAG="--val-data $VAL_DATA"
else
    echo "WARNING: $VAL_DATA not found — 5% of train will be used for validation."
    VAL_FLAG=""
fi
echo "Training data: $TRAIN_DATA"

# ---------------------------------------------------------------------------
# [1/6] Train SSAE c=2 — ReLU + L1, unfrozen encoder, bfloat16
#
# No --topk-k flag  → default ReLU activation + L1 sparsity loss
# No --freeze-encoder → full model trains (same as original SSAE c=1 paper)
# batch-size=32, grad-accum=4 → effective batch 128
# ---------------------------------------------------------------------------
echo "=== [1/6] Training SSAE c=2 ReLU+L1 (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/train_ssae.py \
    --data            "$TRAIN_DATA" \
    ${VAL_FLAG:+--val-data "$VAL_DATA"} \
    --output-dir      "$NEW_CKPT_DIR" \
    --model-id        Qwen/Qwen2.5-0.5B \
    --sparsity-factor 2 \
    --dtype           bfloat16 \
    --epochs          8 \
    --batch-size      32 \
    --grad-accum      4 \
    --lr              1e-4 \
    --min-lr          1e-5 \
    --warmup-steps    500 \
    --num-workers     4 \
    --device          cuda

echo "SSAE c=2 training complete. Checkpoint: $NEW_CKPT"
echo ""

# ---------------------------------------------------------------------------
# [2/6] Encode eval shard
# ---------------------------------------------------------------------------
echo "=== [2/6] Encoding eval shard (GPU 0, offset 0, 90K steps) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/generate_probe_data.py \
    --checkpoint  "$NEW_CKPT" \
    --output      "$C2_EVAL" \
    --offset      0 \
    --max-steps   90000 \
    --batch-size  32 \
    --max-seq-len 2048 \
    --encoding    sparse \
    --device      cuda
echo ""

# ---------------------------------------------------------------------------
# [3/6] Encode train shards in parallel (4 × 90K)
# ---------------------------------------------------------------------------
EVAL_SIZE=90000
SHARD_SIZE=90000
N_SHARDS=4

echo "=== [3/6] Encoding training shards (4 GPUs in parallel) ==="
PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
    OFFSET=$((EVAL_SIZE + i * SHARD_SIZE))
    OUT="$SCRATCH_DATA/c2_train_shard_${i}.npz"
    echo "  GPU $i: offset=$OFFSET → $OUT"
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_probe_data.py \
        --checkpoint  "$NEW_CKPT" \
        --output      "$OUT" \
        --offset      "$OFFSET" \
        --max-steps   "$SHARD_SIZE" \
        --batch-size  32 \
        --max-seq-len 2048 \
        --encoding    sparse \
        --device      cuda \
        > "$SCRATCH_DATA/c2_shard_${i}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  Shard $i OK"
    else
        echo "  ERROR: shard $i failed — see $SCRATCH_DATA/c2_shard_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED shard(s) failed. Aborting."; exit 1; }
echo ""

# ---------------------------------------------------------------------------
# [4/6] Merge shards
# ---------------------------------------------------------------------------
echo "=== [4/6] Merging c=2 training shards ==="
python scripts/slurm/merge_shards.py \
    --inputs  $(for i in $(seq 0 $((N_SHARDS-1))); do echo "$SCRATCH_DATA/c2_train_shard_${i}.npz"; done) \
    --output  "$C2_TRAIN"
echo ""

# ---------------------------------------------------------------------------
# [5/6] Train probes — 4 seeds in parallel
# ---------------------------------------------------------------------------
SEEDS=(42 43 44 45)

echo "=== [5/6a] Training c=2 linear probes ==="
PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    CUDA_VISIBLE_DEVICES=$i python scripts/experiment_linear_probe.py \
        --train-data "$C2_TRAIN" \
        --eval-data  "$C2_EVAL" \
        --output     "$SCRATCH_RESULTS/c2_linear_probe_seed${SEED}.pt" \
        --seed       "$SEED" \
        --epochs     50 \
        --batch-size 512 \
        --device     cuda \
        > "$SCRATCH_RESULTS/c2_linear_probe_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" && echo "  Linear probe seed ${SEEDS[$i]} OK" \
        || { echo "  ERROR: linear probe seed ${SEEDS[$i]} failed"; FAILED=$((FAILED+1)); }
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED linear probe(s) failed."; exit 1; }
echo ""

echo "=== [5/6b] Training c=2 MLP probes ==="
PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    CUDA_VISIBLE_DEVICES=$i python scripts/experiment_full_clean.py \
        --train-data "$C2_TRAIN" \
        --eval-data  "$C2_EVAL" \
        --output     "$SCRATCH_RESULTS/c2_probe_seed${SEED}.pt" \
        --seed       "$SEED" \
        --epochs     50 \
        --batch-size 512 \
        --device     cuda \
        > "$SCRATCH_RESULTS/c2_probe_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
done
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" && echo "  MLP probe seed ${SEEDS[$i]} OK" \
        || { echo "  ERROR: MLP probe seed ${SEEDS[$i]} failed"; FAILED=$((FAILED+1)); }
done
[ "$FAILED" -gt 0 ] && { echo "ERROR: $FAILED MLP probe(s) failed."; exit 1; }
echo ""

# ---------------------------------------------------------------------------
# [6/6] Threshold sweep
# ---------------------------------------------------------------------------
echo "=== [6/6] Evaluating c=2 probes ==="
python - <<PYEOF | tee "$PROJECT_DIR/results/c2_probe_eval.txt"
import numpy as np, torch, torch.nn as nn, statistics
from pathlib import Path

results_dir = Path("$SCRATCH_RESULTS")
eval_path   = "$C2_EVAL"

def load_eval(path, seed=42):
    d = np.load(path)
    h, y = d["latents"].astype(np.float32), d["correctness"].astype(np.int64)
    rng = np.random.default_rng(seed)
    cor = np.where(y==1)[0]; inc = np.where(y==0)[0]
    n = min(len(cor), len(inc), 25000)
    sel = np.concatenate([rng.choice(cor,n,replace=False), rng.choice(inc,n,replace=False)])
    rng.shuffle(sel)
    print(f"  Eval: {len(sel):,} steps from {path}")
    return h[sel], y[sel]

def eval_probe(model, h, y, t):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(h)).squeeze(-1)
        pred = (torch.sigmoid(logits) >= t).long().numpy()
    acc = (pred==y).mean()
    def f1(c):
        tp=((pred==c)&(y==c)).sum(); fp=((pred==c)&(y!=c)).sum(); fn=((pred!=c)&(y==c)).sum()
        p=tp/(tp+fp) if (tp+fp) else 0.0; r=tp/(tp+fn) if (tp+fn) else 0.0
        return 2*p*r/(p+r) if (p+r) else 0.0
    return acc, f1(1), f1(0)

class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h,h//2), nn.LayerNorm(h//2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h//2,1))
    def forward(self,x): return self.net(x)

class Linear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d,1)
    def forward(self,x): return self.fc(x)

h_eval, y_eval = load_eval(eval_path)
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
seeds = [42, 43, 44, 45]

for ptype, prefix, loader_fn in [
    ("MLP",    "c2_probe_seed",        lambda p: (lambda ckpt, cfg: (m:=MLP(cfg["input_dim"],cfg["hidden_dim"]), m.load_state_dict(ckpt["model"]), m)[2])(*(lambda c: (c, c["config"]))(torch.load(p,map_location="cpu",weights_only=False)))),
    ("Linear", "c2_linear_probe_seed", lambda p: (lambda sd: (m:=Linear(sd["fc.weight"].shape[1]), m.load_state_dict(sd), m)[2])(torch.load(p,map_location="cpu",weights_only=True)["state_dict"])),
]:
    print(f"\n{'='*60}\n  c=2 ReLU+L1 {ptype} probe — threshold sweep\n{'='*60}")
    pt = {t: [] for t in thresholds}
    for seed in seeds:
        path = results_dir / f"{prefix}{seed}.pt"
        if not path.exists(): print(f"  WARNING: {path} missing"); continue
        model = loader_fn(str(path))
        for t in thresholds:
            acc, f1c, f1i = eval_probe(model, h_eval, y_eval, t)
            pt[t].append((acc, f1c, f1i, (f1c+f1i)/2))
            print(f"  seed={seed}  t={t:.1f}  acc={acc*100:.2f}%  macro={(f1c+f1i)/2:.3f}")
    print(f"\n  Mean ± std:  t     acc            macro_f1")
    for t in thresholds:
        rows = pt[t]
        if not rows: continue
        accs=[r[0]*100 for r in rows]; macros=[r[3] for r in rows]
        am=statistics.mean(accs); as_=statistics.stdev(accs) if len(accs)>1 else 0.0
        mm=statistics.mean(macros); ms_=statistics.stdev(macros) if len(macros)>1 else 0.0
        print(f"  {t:.1f}   {am:.2f}±{as_:.2f}%   {mm:.3f}±{ms_:.3f}")
PYEOF

# ---------------------------------------------------------------------------
# Save checkpoint + results to persistent storage
# ---------------------------------------------------------------------------
mkdir -p "$CKPT_DIR" "$STORE/results_c2"
cp "$NEW_CKPT" "$CKPT_DIR/ssae_c2_relu_l1.pt" 2>/dev/null || true
cp "$SCRATCH_RESULTS"/c2_linear_probe_seed*.{pt,log} "$STORE/results_c2/" 2>/dev/null || true
cp "$SCRATCH_RESULTS"/c2_probe_seed*.{pt,log}        "$STORE/results_c2/" 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "  Checkpoint : $CKPT_DIR/ssae_c2_relu_l1.pt"
echo "  Probes     : $STORE/results_c2/"
echo ""
echo "Next step — submit the comparison job:"
echo "  sbatch scripts/slurm/tamia_c2_encode_and_compare.sh"
