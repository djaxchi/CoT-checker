#!/usr/bin/env python3
"""Linear-probe baseline on SSAE latents.

Drops the MLP in `experiment_full_clean.py` and trains a single Linear(896 -> 1)
layer on the same 70/30 subsampled training data with the same held-out 50K
balanced eval. Same SUMMARY line format so existing aggregation works.

The mechanistic claim: if this matches the MLP, the correctness signal is
linearly decodable from h_c. If it falls short, the structure is non-linear.

Usage:
    python scripts/experiment_linear_probe.py \\
        --train-data $SCRATCH/cot-checker/probe_data/train_final.npz \\
        --eval-data  $SCRATCH/cot-checker/probe_data/eval_held_out.npz \\
        --output     results/probes/linear_probe_seed42.pt \\
        --seed       42 \\
        --device     cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Reuse helpers from the MLP experiment to keep both runs identical where possible.
from scripts.experiment_full_clean import (  # type: ignore
    ArrayDataset,
    build_training_subset,
    evaluate,
    benchmark_latency,
    print_results,
    train_epoch,
)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

    def save(self, path: Path):
        torch.save({"state_dict": self.state_dict(), "input_dim": self.fc.in_features}, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--output", default="results/probes/linear_probe.pt")
    p.add_argument("--lr", type=float, default=1e-3)  # linear probes like higher lr
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--split", type=float, default=0.9)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--correct-frac", type=float, default=0.70)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # [1/4] Load data
    print(f"\n[1/4] Loading training data from {args.train_data}...")
    d = np.load(args.train_data)
    h, y = d["latents"].astype(np.float32), d["correctness"]
    total = len(y)
    n_pos = int((y == 1).sum())
    print(f"  Total steps  : {total:,}")
    print(f"  Correct (+)  : {n_pos:,}  ({n_pos/total:.1%})")
    print(f"  Incorrect (-): {total - n_pos:,}  ({(total - n_pos)/total:.1%})")
    print(f"  Latent dim   : {h.shape[1]}")

    if args.correct_frac > 0:
        print(f"\n  Subsampling to {args.correct_frac:.0%} correct / {1-args.correct_frac:.0%} incorrect...")
        h, y = build_training_subset(h, y, args.correct_frac, args.seed)

    full_ds = ArrayDataset(h, y)
    n_train = int(len(full_ds) * args.split)
    train_ds, val_ds = random_split(
        full_ds, [n_train, len(full_ds) - n_train],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train split  : {len(train_ds):,}")
    print(f"  Internal val : {len(val_ds):,}")

    # [2/4] Train linear probe
    print(f"\n[2/4] Training linear probe ({args.epochs} epochs, batch={args.batch_size})...")
    model = LinearProbe(input_dim=h.shape[1]).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Probe parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(args.device == "cuda"),
    )

    best_acc, best_state = 0.0, None
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Acc':>8}")
    print("-" * 30)

    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, args.device, pos_weight=None)
        val_acc, _ = evaluate(model, val_loader, args.device)
        scheduler.step()
        marker = " *" if val_acc > best_acc else ""
        print(f"{epoch:>5}  {loss:>10.4f}  {val_acc:>8.4f}{marker}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_wall_sec = time.perf_counter() - train_start

    model.load_state_dict(best_state)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"\n  Training time : {train_wall_sec:.1f}s total")
    print(f"  Best internal val acc: {best_acc*100:.2f}%")
    print(f"  Saved -> {out}")

    # [3/4] Held-out eval
    print(f"\n[3/4] Held-out evaluation (threshold={args.threshold})...")
    eval_d = np.load(args.eval_data)
    eval_h = eval_d["latents"].astype(np.float32)
    eval_y = eval_d["correctness"]

    rng = np.random.default_rng(args.seed)
    cor_idx = np.where(eval_y == 1)[0]
    inc_idx = np.where(eval_y == 0)[0]
    n_per_class = min(len(cor_idx), len(inc_idx), 25000)
    cor_sel = rng.choice(cor_idx, size=n_per_class, replace=False)
    inc_sel = rng.choice(inc_idx, size=n_per_class, replace=False)
    sel = np.concatenate([cor_sel, inc_sel])
    rng.shuffle(sel)
    eval_h, eval_y = eval_h[sel], eval_y[sel]
    print(f"  Balanced subset: {len(eval_y):,} steps (50/50)")

    eval_ds = ArrayDataset(eval_h, eval_y)
    eval_loader = DataLoader(eval_ds, batch_size=512, shuffle=False, num_workers=2)

    acc, results = evaluate(model, eval_loader, args.device, threshold=args.threshold)
    print_results(
        f"Held-out eval  |  LINEAR  seed={args.seed}  threshold={args.threshold}",
        acc, results, majority=0.5,
    )

    print("Threshold sweep:")
    print(f"  {'Threshold':>10}  {'Accuracy':>10}  {'Correct F1':>12}  {'Incorrect F1':>13}  {'Macro F1':>10}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        a, r = evaluate(model, eval_loader, args.device, threshold=t)
        macro_f1 = (r["correct"]["f1"] + r["incorrect"]["f1"]) / 2
        print(f"  {t:>10.1f}  {a*100:>9.2f}%  {r['correct']['f1']:>12.3f}  {r['incorrect']['f1']:>13.3f}  {macro_f1:>10.3f}")

    # [4/4] Latency
    print(f"\n[4/4] Inference latency benchmark (device={args.device})...")
    lat = benchmark_latency(model, h.shape[1], args.device)
    print(f"  Single-step: mean={lat['mean_ms']:.3f}ms  p99={lat['p99_ms']:.3f}ms")
    print(f"  Batch (512): {lat['batch512_ms']:.3f}ms  throughput={lat['throughput_per_sec']:,.0f} steps/sec")

    print(
        f"\nSUMMARY model=linear seed={args.seed}"
        f" acc={acc*100:.2f}"
        f" gain={(acc - 0.5)*100:.2f}"
        f" f1_correct={results['correct']['f1']:.3f}"
        f" f1_incorrect={results['incorrect']['f1']:.3f}"
        f" macro_f1={(results['correct']['f1']+results['incorrect']['f1'])/2:.3f}"
        f" train_sec={train_wall_sec:.1f}"
        f" latency_mean_ms={lat['mean_ms']:.3f}"
        f" latency_p99_ms={lat['p99_ms']:.3f}"
        f" throughput_per_sec={lat['throughput_per_sec']:.0f}"
    )


if __name__ == "__main__":
    main()
