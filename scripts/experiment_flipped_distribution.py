#!/usr/bin/env python3
"""Flipped-distribution probe experiment.

Tests the calibration hypothesis: if the 50/50 probe underperforms because
the classifier learns the wrong prior, then flipping to 72% correct / 28%
incorrect (the mirror of the natural distribution) should hurt in the
opposite direction -- over-predicting correct, missing incorrect steps.

Builds a ~40K step training set at 72/28 (correct/incorrect) by oversampling
the minority correct class and subsampling incorrect, then trains and evaluates
using the same hyperparameters as the 40K natural experiment.

Evaluation is on the shared 5K held-out Math-Shepherd set.

Usage:
    python scripts/experiment_flipped_distribution.py --device mps
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_flipped_dataset(src_npz: str, target_total: int, correct_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (latents, labels) resampled to target_total with correct_frac positive."""
    rng = np.random.default_rng(seed)
    d = np.load(src_npz)
    h, y = d["latents"].astype(np.float32), d["correctness"]

    correct_idx   = np.where(y == 1)[0]
    incorrect_idx = np.where(y == 0)[0]

    n_correct   = int(target_total * correct_frac)
    n_incorrect = target_total - n_correct

    # Oversample correct (with replacement if needed), subsample incorrect
    sampled_correct   = rng.choice(correct_idx,   size=n_correct,   replace=len(correct_idx)   < n_correct)
    sampled_incorrect = rng.choice(incorrect_idx, size=n_incorrect, replace=len(incorrect_idx) < n_incorrect)

    idx = np.concatenate([sampled_correct, sampled_incorrect])
    rng.shuffle(idx)

    print(f"  Flipped dataset: {n_correct} correct ({correct_frac*100:.0f}%), {n_incorrect} incorrect ({(1-correct_frac)*100:.0f}%)")
    print(f"  Correct oversampled: {len(correct_idx)} unique -> {n_correct} (x{n_correct/len(correct_idx):.2f})")
    print(f"  Incorrect subsampled: {len(incorrect_idx)} -> {n_incorrect}")
    return h[idx], y[idx]


class ArrayDataset(Dataset):
    def __init__(self, latents: np.ndarray, labels: np.ndarray):
        self.latents     = torch.from_numpy(latents)
        self.correctness = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.correctness[idx]


class NpzDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.latents     = torch.from_numpy(d["latents"].astype(np.float32))
        self.correctness = torch.from_numpy(d["correctness"].astype(np.float32))

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.correctness[idx]


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
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
    for cls, cls_name in [(1, "correct"), (0, "incorrect")]:
        tp = ((preds_t == cls) & (labels_t == cls)).sum().item()
        fp = ((preds_t == cls) & (labels_t != cls)).sum().item()
        fn = ((preds_t != cls) & (labels_t == cls)).sum().item()
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results[cls_name] = {"precision": p, "recall": r, "f1": f1}

    return acc, results


def train_probe(train_ds, val_ds, n_latents, args):
    model = StepCorrectnessClassifier(
        input_dim=n_latents,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    best_acc, best_state = 0.0, None
    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}")
    print("─" * 28)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, args.device)
        val_acc, _ = evaluate(model, val_loader, args.device)
        scheduler.step()
        marker = " *" if val_acc > best_acc else ""
        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>8.4f}{marker}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"\nBest val acc: {best_acc*100:.2f}%")
    return model


def print_results(label, acc, results, majority):
    print(f"\n{'─'*52}")
    print(f"  {label}")
    print(f"{'─'*52}")
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
    p.add_argument("--src",        default="results/probe_data/math_shepherd_100k_natural.npz")
    p.add_argument("--eval-data",  default="results/probe_data/math_shepherd_eval_5000.npz")
    p.add_argument("--natural-probe", default="results/probes/correctness_probe_100k_natural.pt",
                   help="Already-trained 40K natural probe to compare against")
    p.add_argument("--output",     default="results/probes/correctness_probe_flipped.pt")
    p.add_argument("--target-total", type=int, default=40000)
    p.add_argument("--correct-frac", type=float, default=0.72,
                   help="Fraction of correct steps in flipped dataset (default: 0.72)")
    p.add_argument("--hidden-dim", type=int,   default=1024)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--split",      type=float, default=0.8)
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # --- Build flipped dataset ---
    print(f"\nBuilding flipped dataset ({args.correct_frac*100:.0f}% correct / {(1-args.correct_frac)*100:.0f}% incorrect)...")
    h_flip, y_flip = build_flipped_dataset(args.src, args.target_total, args.correct_frac, args.seed)

    full_ds = ArrayDataset(h_flip, y_flip)
    n_train = int(len(full_ds) * args.split)
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    n_latents = h_flip.shape[1]
    print(f"  Train: {n_train}  Val: {n_val}  Latent dim: {n_latents}")

    # --- Train flipped probe ---
    print("\nTraining flipped probe...")
    model = train_probe(train_ds, val_ds, n_latents, args)
    model.save(Path(args.output))
    print(f"Saved to {args.output}")

    # --- Evaluate both probes on shared 5K held-out set ---
    eval_ds     = NpzDataset(args.eval_data)
    eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False)
    eval_majority = max(eval_ds.correctness.mean().item(), 1 - eval_ds.correctness.mean().item())

    print("\n\n=== HELD-OUT EVALUATION (5K Math-Shepherd steps) ===")

    # Flipped probe
    acc_flip, res_flip = evaluate(model, eval_loader, args.device)
    print_results(f"Flipped probe ({args.correct_frac*100:.0f}% correct / {(1-args.correct_frac)*100:.0f}% incorrect train dist.)",
                  acc_flip, res_flip, eval_majority)

    # Natural probe (already trained)
    if Path(args.natural_probe).exists():
        nat_model = StepCorrectnessClassifier.load(args.natural_probe, device=args.device)
        acc_nat, res_nat = evaluate(nat_model, eval_loader, args.device)
        print_results("Natural probe (28% correct / 72% incorrect train dist.)",
                      acc_nat, res_nat, eval_majority)

        # Summary comparison
        print(f"\n{'─'*52}")
        print("  SUMMARY COMPARISON")
        print(f"{'─'*52}")
        print(f"  {'':30s} {'Natural':>8}  {'Flipped':>8}")
        print(f"  {'Accuracy':30s} {acc_nat*100:>8.2f}%  {acc_flip*100:>8.2f}%")
        for cls in ["correct", "incorrect"]:
            for metric in ["precision", "recall", "f1"]:
                label = f"{cls} {metric}"
                print(f"  {label:30s} {res_nat[cls][metric]:>8.3f}   {res_flip[cls][metric]:>8.3f}")
    else:
        print(f"\nNatural probe not found at {args.natural_probe}, skipping comparison.")


if __name__ == "__main__":
    main()
