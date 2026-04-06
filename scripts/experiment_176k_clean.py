#!/usr/bin/env python3
"""Experiment: 176K natural data, 70/30 subset, no duplication.

Uses the 176K steps encoded at max_len=128 (math_shepherd_176k_natural.npz).
Subsamples to 70% correct / 30% incorrect WITHOUT any oversampling —
both classes are subsets of the unique encoded steps.

Pool:
  correct  :  50,950 unique steps  → keep all
  incorrect: 125,058 unique steps  → subsample to int(50950 * 30/70) ≈ 21,836

Total training set: ~72,786 steps (no duplicates).

Evaluates on the clean held-out set (math_shepherd_eval_5k_clean.npz)
which is from offset 100K, a gap between the two training windows.

Usage:
    python scripts/experiment_176k_clean.py --device mps
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.probes.classifier import StepCorrectnessClassifier

TRAIN_DATA = Path("results/probe_data/math_shepherd_176k_natural.npz")
EVAL_DATA  = Path("results/probe_data/math_shepherd_eval_5k_clean.npz")
PROBE_OUT  = Path("results/probes/correctness_probe_176k_clean.pt")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class ArrayDataset(Dataset):
    def __init__(self, h: np.ndarray, y: np.ndarray):
        self.latents     = torch.from_numpy(h)
        self.correctness = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, i):
        return self.latents[i], self.correctness[i]


def build_subset(npz_path: str, correct_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Subsample npz to correct_frac / (1-correct_frac) with NO duplication."""
    rng = np.random.default_rng(seed)
    d = np.load(npz_path)
    h, y = d["latents"].astype(np.float32), d["correctness"]

    cor_idx = np.where(y == 1)[0]
    inc_idx = np.where(y == 0)[0]

    # Minority class drives the total — keep all of it, subsample majority
    n_cor = len(cor_idx)
    n_inc = len(inc_idx)
    minority_is_correct = n_cor < n_inc

    if minority_is_correct:
        # Keep all correct, subsample incorrect to match ratio
        n_inc_keep = int(n_cor * (1 - correct_frac) / correct_frac)
        n_inc_keep = min(n_inc_keep, n_inc)
        sampled_cor = cor_idx
        sampled_inc = rng.choice(inc_idx, size=n_inc_keep, replace=False)
    else:
        n_cor_keep = int(n_inc * correct_frac / (1 - correct_frac))
        n_cor_keep = min(n_cor_keep, n_cor)
        sampled_cor = rng.choice(cor_idx, size=n_cor_keep, replace=False)
        sampled_inc = inc_idx

    idx = np.concatenate([sampled_cor, sampled_inc])
    rng.shuffle(idx)
    h_out, y_out = h[idx], y[idx]

    total = len(y_out)
    pos   = (y_out == 1).sum()
    print(f"  Subset: {total:,} steps  "
          f"({pos:,} correct {pos/total:.1%}, {total-pos:,} incorrect {(total-pos)/total:.1%})")
    print(f"  No duplication — correct unique: {len(sampled_cor):,}, incorrect unique: {len(sampled_inc):,}")
    return h_out, y_out


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
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
    preds_t  = torch.tensor(all_preds)
    acc = (preds_t == labels_t).float().mean().item()

    results = {}
    for cls, name in [(1, "correct"), (0, "incorrect")]:
        tp = ((preds_t == cls) & (labels_t == cls)).sum().item()
        fp = ((preds_t == cls) & (labels_t != cls)).sum().item()
        fn = ((preds_t != cls) & (labels_t == cls)).sum().item()
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results[name] = {"precision": p, "recall": r, "f1": f1}
    return acc, results


def print_results(label, acc, results, majority):
    print(f"\n{'─'*54}")
    print(f"  {label}")
    print(f"{'─'*54}")
    print(f"  Majority baseline : {majority*100:.2f}%")
    print(f"  Accuracy          : {acc*100:.2f}%  (+{(acc-majority)*100:.2f}pp)")
    for cls in ["correct", "incorrect"]:
        r = results[cls]
        print(f"\n  Class '{cls}':")
        print(f"    Precision : {r['precision']:.3f}")
        print(f"    Recall    : {r['recall']:.3f}")
        print(f"    F1        : {r['f1']:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data",   default=str(TRAIN_DATA))
    p.add_argument("--eval-data",    default=str(EVAL_DATA))
    p.add_argument("--output",       default=str(PROBE_OUT))
    p.add_argument("--correct-frac", type=float, default=0.70,
                   help="Fraction of correct steps in training subset (default: 0.70)")
    p.add_argument("--hidden-dim",   type=int,   default=1024)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--split",        type=float, default=0.8)
    p.add_argument("--threshold",    type=float, default=0.8)
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # --- Build subset ---
    print(f"\n[1/3] Building {args.correct_frac:.0%}/{1-args.correct_frac:.0%} subset from {args.train_data}...")
    h, y = build_subset(args.train_data, args.correct_frac, args.seed)

    full_ds = ArrayDataset(h, y)
    n_train = int(len(full_ds) * args.split)
    train_ds, val_ds = random_split(
        full_ds, [n_train, len(full_ds) - n_train],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Latent dim: {h.shape[1]}")

    # --- Train ---
    print(f"\n[2/3] Training probe ({args.epochs} epochs)...")
    model = StepCorrectnessClassifier(
        input_dim=h.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    best_acc, best_state = 0.0, None
    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}")
    print("─" * 28)
    for epoch in range(1, args.epochs + 1):
        loss    = train_epoch(model, train_loader, optimizer, args.device)
        val_acc, _ = evaluate(model, val_loader, args.device)
        scheduler.step()
        marker = " *" if val_acc > best_acc else ""
        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>8.4f}{marker}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"\n  Best val acc: {best_acc*100:.2f}%  |  Saved -> {out}")

    # --- Evaluate ---
    print(f"\n[3/3] Evaluating on clean held-out set (threshold={args.threshold})...")
    if not Path(args.eval_data).exists():
        print(f"  ERROR: eval data not found at {args.eval_data}")
        print("  Run generate_probe_data.py with --offset 100000 --max-steps 5000 first.")
        return

    eval_d   = np.load(args.eval_data)
    eval_ds  = ArrayDataset(eval_d["latents"].astype(np.float32), eval_d["correctness"])
    loader   = DataLoader(eval_ds, batch_size=256, shuffle=False)
    majority = max(eval_ds.correctness.mean().item(), 1 - eval_ds.correctness.mean().item())

    acc, results = evaluate(model, loader, args.device, threshold=args.threshold)
    print_results(
        f"176K clean probe (70/30, no duplication, threshold={args.threshold})",
        acc, results, majority
    )


if __name__ == "__main__":
    main()
