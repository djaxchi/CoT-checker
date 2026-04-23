#!/usr/bin/env python3
"""Full-scale clean experiment: MLP probe on complete Math-Shepherd latents.

Trains on the full natural-distribution dataset (no rebalancing by default)
and evaluates on a clean held-out set that has zero overlap with training data.

Key differences from experiment_176k_clean.py:
  - No artificial rebalancing — uses the true ~28/72 correct/incorrect distribution
  - Weighted loss to handle class imbalance without discarding data
  - Larger default batch size for GPU efficiency
  - Full dataset (all available steps)
  - Measures and reports training wall-clock time and probe inference latency

Usage:
    python scripts/experiment_full_clean.py \\
        --train-data $SCRATCH/cot-checker/probe_data/train_full.npz \\
        --eval-data  $SCRATCH/cot-checker/probe_data/eval_held_out.npz \\
        --output     results/probes/probe_full_clean.pt \\
        --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.probes.classifier import StepCorrectnessClassifier


class ArrayDataset(Dataset):
    def __init__(self, h: np.ndarray, y: np.ndarray):
        self.latents = torch.from_numpy(h.astype(np.float32))
        self.correctness = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, i):
        return self.latents[i], self.correctness[i]


def build_training_subset(h: np.ndarray, y: np.ndarray, correct_frac: float, seed: int):
    """Subsample to correct_frac / (1-correct_frac) without duplication.

    Keeps all minority-class samples intact and subsamples the majority.
    With correct_frac=0.70: keeps all correct steps, subsamples incorrect
    to 30/70 of the correct count. This replicates the balance used in the
    previous best run (73.4% accuracy, section 7.1 of REPORT.md).
    """
    rng = np.random.default_rng(seed)
    cor_idx = np.where(y == 1)[0]
    inc_idx = np.where(y == 0)[0]

    n_cor = len(cor_idx)
    n_inc = len(inc_idx)
    minority_is_correct = n_cor < n_inc

    if minority_is_correct:
        n_inc_keep = int(n_cor * (1 - correct_frac) / correct_frac)
        n_inc_keep = min(n_inc_keep, n_inc)
        sel_cor = cor_idx
        sel_inc = rng.choice(inc_idx, size=n_inc_keep, replace=False)
    else:
        n_cor_keep = int(n_inc * correct_frac / (1 - correct_frac))
        n_cor_keep = min(n_cor_keep, n_cor)
        sel_cor = rng.choice(cor_idx, size=n_cor_keep, replace=False)
        sel_inc = inc_idx

    idx = np.concatenate([sel_cor, sel_inc])
    rng.shuffle(idx)
    h_out, y_out = h[idx], y[idx]

    pos = int((y_out == 1).sum())
    total = len(y_out)
    print(f"  Subset: {total:,} steps  ({pos:,} correct {pos/total:.1%} / {total-pos:,} incorrect {(total-pos)/total:.1%})")
    print(f"  Correct unique: {len(sel_cor):,}  Incorrect unique: {len(sel_inc):,}  (no duplication)")
    return h_out, y_out


def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.binary_cross_entropy_with_logits(
            model(x).squeeze(-1), y, pos_weight=pos_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = (torch.sigmoid(model(x).squeeze(-1)) >= threshold).long()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.long().cpu().tolist())

    labels_t = torch.tensor(all_labels)
    preds_t = torch.tensor(all_preds)
    acc = (preds_t == labels_t).float().mean().item()

    results = {}
    for cls, name in [(1, "correct"), (0, "incorrect")]:
        tp = ((preds_t == cls) & (labels_t == cls)).sum().item()
        fp = ((preds_t == cls) & (labels_t != cls)).sum().item()
        fn = ((preds_t != cls) & (labels_t == cls)).sum().item()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results[name] = {"precision": p, "recall": r, "f1": f1}
    return acc, results


@torch.no_grad()
def benchmark_latency(model, latent_dim: int, device: str, n_warmup=50, n_trials=1000):
    """Measure single-step probe inference latency.

    Runs n_warmup forward passes to warm up the GPU, then times n_trials
    individual single-sample forward passes. Reports mean, std, p50, p99 in ms,
    and throughput at batch_size=512.
    """
    model.eval()
    dummy = torch.randn(1, latent_dim, device=device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # Single-step latency
    times_ms = []
    for _ in range(n_trials):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)

    # Batch throughput
    batch = torch.randn(512, latent_dim, device=device)
    for _ in range(n_warmup):
        _ = model(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = model(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    batch_ms = (time.perf_counter() - t0) * 1000 / 100  # avg ms per batch of 512

    return {
        "mean_ms": times_ms.mean(),
        "std_ms": times_ms.std(),
        "p50_ms": np.percentile(times_ms, 50),
        "p99_ms": np.percentile(times_ms, 99),
        "batch512_ms": batch_ms,
        "throughput_per_sec": 512 / (batch_ms / 1000),
    }


def print_results(label, acc, results, majority):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Majority baseline : {majority*100:.2f}%")
    print(f"  Accuracy          : {acc*100:.2f}%  (+{(acc-majority)*100:.2f}pp)")
    for cls in ["correct", "incorrect"]:
        r = results[cls]
        print(f"\n  Class '{cls}':")
        print(f"    Precision : {r['precision']:.3f}")
        print(f"    Recall    : {r['recall']:.3f}")
        print(f"    F1        : {r['f1']:.3f}")
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--output", default="results/probes/probe_full_clean.pt")
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--split", type=float, default=0.9,
                   help="Fraction of training data used for training (rest = internal val)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for the final held-out eval")
    p.add_argument("--correct-frac", type=float, default=0.70,
                   help="Fraction of correct steps in training subset. "
                        "Default 0.70 replicates the best previous run (REPORT.md §7.1). "
                        "Set to -1 to use the full natural distribution with weighted loss.")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------------------------------------------------------
    # [1/4] Load training data
    # -------------------------------------------------------------------------
    print(f"\n[1/4] Loading training data from {args.train_data}...")
    d = np.load(args.train_data)
    h, y = d["latents"].astype(np.float32), d["correctness"]

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    total = len(y)
    print(f"  Total steps  : {total:,}")
    print(f"  Correct (+)  : {n_pos:,}  ({n_pos/total:.1%})")
    print(f"  Incorrect (-): {n_neg:,}  ({n_neg/total:.1%})")
    print(f"  Latent dim   : {h.shape[1]}")

    # --- Subsample to target training distribution ---
    if args.correct_frac > 0:
        print(f"\n  Subsampling to {args.correct_frac:.0%} correct / {1-args.correct_frac:.0%} incorrect...")
        h, y = build_training_subset(h, y, args.correct_frac, args.seed)
    else:
        print(f"\n  Using full natural distribution with weighted loss.")
        n_pos_nat = int((y == 1).sum())
        print(f"  Natural: {len(y):,} steps  ({n_pos_nat/len(y):.1%} correct)")

    full_ds = ArrayDataset(h, y)
    n_train = int(len(full_ds) * args.split)
    train_ds, val_ds = random_split(
        full_ds, [n_train, len(full_ds) - n_train],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train split  : {len(train_ds):,}")
    print(f"  Internal val : {len(val_ds):,}")

    # Use weighted loss only when not explicitly subsampling
    pos_weight = None
    if args.correct_frac <= 0:
        n_pos = int((y == 1).sum())
        w = (len(y) - n_pos) / max(n_pos, 1)
        pos_weight = torch.tensor([w], device=args.device)
        print(f"  pos_weight (correct class): {w:.3f}")

    # -------------------------------------------------------------------------
    # [2/4] Train
    # -------------------------------------------------------------------------
    print(f"\n[2/4] Training probe ({args.epochs} epochs, batch={args.batch_size})...")
    model = StepCorrectnessClassifier(
        input_dim=h.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(args.device)

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
    print("─" * 30)

    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, args.device, pos_weight)
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

    m, s = divmod(int(train_wall_sec), 60)
    h_t, m_t = divmod(m, 60)
    print(f"\n  Training time : {h_t:02d}h {m_t:02d}m {s:02d}s  ({train_wall_sec:.1f}s total)")
    print(f"  Best internal val acc: {best_acc*100:.2f}%")
    print(f"  Saved -> {out}")

    # -------------------------------------------------------------------------
    # [3/4] Held-out evaluation (balanced 50/50 subset)
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Held-out evaluation (threshold={args.threshold})...")
    eval_d = np.load(args.eval_data)
    eval_h = eval_d["latents"].astype(np.float32)
    eval_y = eval_d["correctness"]

    # Subsample to 50/50, capped at 25K per class (50K total).
    rng = np.random.default_rng(args.seed)
    cor_idx = np.where(eval_y == 1)[0]
    inc_idx = np.where(eval_y == 0)[0]
    n_per_class = min(len(cor_idx), len(inc_idx), 25000)
    cor_sel = rng.choice(cor_idx, size=n_per_class, replace=False)
    inc_sel = rng.choice(inc_idx, size=n_per_class, replace=False)
    sel = np.concatenate([cor_sel, inc_sel])
    rng.shuffle(sel)
    eval_h, eval_y = eval_h[sel], eval_y[sel]

    eval_n_pos = int((eval_y == 1).sum())
    eval_n_neg = int((eval_y == 0).sum())
    eval_total = len(eval_y)
    print(f"  Raw held-out   : {len(eval_d['correctness']):,} steps (natural distribution)")
    print(f"  Balanced subset: {eval_total:,} steps ({eval_n_pos:,} correct / {eval_n_neg:,} incorrect)")
    print(f"  Majority baseline: 50.0%  (by construction)")

    eval_majority = 0.5
    eval_ds = ArrayDataset(eval_h, eval_y)
    eval_loader = DataLoader(eval_ds, batch_size=512, shuffle=False, num_workers=2)

    acc, results = evaluate(model, eval_loader, args.device, threshold=args.threshold)
    print_results(
        f"Held-out eval  |  seed={args.seed}  threshold={args.threshold}",
        acc, results, eval_majority,
    )

    print("Threshold sweep:")
    print(f"  {'Threshold':>10}  {'Accuracy':>10}  {'Correct F1':>12}  {'Incorrect F1':>13}  {'Macro F1':>10}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        a, r = evaluate(model, eval_loader, args.device, threshold=t)
        macro_f1 = (r["correct"]["f1"] + r["incorrect"]["f1"]) / 2
        print(
            f"  {t:>10.1f}  {a*100:>9.2f}%"
            f"  {r['correct']['f1']:>12.3f}"
            f"  {r['incorrect']['f1']:>13.3f}"
            f"  {macro_f1:>10.3f}"
        )

    # -------------------------------------------------------------------------
    # [4/4] Inference latency benchmark
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Inference latency benchmark (device={args.device})...")
    lat = benchmark_latency(model, h.shape[1], args.device)
    print(f"  Single-step latency (n=1000 trials):")
    print(f"    Mean : {lat['mean_ms']:.3f} ms")
    print(f"    Std  : {lat['std_ms']:.3f} ms")
    print(f"    p50  : {lat['p50_ms']:.3f} ms")
    print(f"    p99  : {lat['p99_ms']:.3f} ms")
    print(f"  Batch throughput (batch=512):")
    print(f"    Latency    : {lat['batch512_ms']:.3f} ms per batch")
    print(f"    Throughput : {lat['throughput_per_sec']:,.0f} steps/sec")

    # Machine-readable summary line for easy grep/aggregation
    print(
        f"\nSUMMARY seed={args.seed}"
        f" acc={acc*100:.2f}"
        f" gain={( acc - eval_majority)*100:.2f}"
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
