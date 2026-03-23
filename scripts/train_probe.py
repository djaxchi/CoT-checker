#!/usr/bin/env python3
"""Step 2 — Train the step-correctness MLP probe on SSAE latents.

Reads the .npz produced by generate_probe_data.py and trains a 3-layer MLP
to predict step correctness from the sparse latent vector h_c.

Usage:
    python scripts/train_probe.py \\
        --train results/probe_data/gsm8k_train.npz \\
        --val   results/probe_data/gsm8k_valid.npz \\
        --output results/probes/correctness_probe.pt

    # Quick test with a single .npz used for both train and val:
    python scripts/train_probe.py \\
        --train results/probe_data/gsm8k_valid_small.npz \\
        --val   results/probe_data/gsm8k_valid_small.npz \\
        --epochs 5 --batch-size 64
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
# Dataset
# ---------------------------------------------------------------------------


class LatentsDataset(Dataset):
    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)
        self.latents = torch.from_numpy(data["latents"].astype(np.float32))
        self.correctness = torch.from_numpy(data["correctness"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.correctness[idx]


# ---------------------------------------------------------------------------
# Training
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
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(-1)
        preds = (torch.sigmoid(logits) >= threshold).long()
        correct += (preds == y.long()).sum().item()
        total += len(y)
    return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Train step-correctness probe")
    # Either provide separate train/val files, or a single file with --split
    p.add_argument("--train", default=None, help=".npz file for training")
    p.add_argument("--val", default=None, help=".npz file for validation")
    p.add_argument("--data", default=None, help="Single .npz file — will be split by --split ratio")
    p.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Fraction used for training when --data is given (default 0.8)",
    )
    p.add_argument("--output", default="results/probes/correctness_probe.pt")
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    if args.data:
        full_ds = LatentsDataset(args.data)
        n_train = int(len(full_ds) * args.split)
        n_val = len(full_ds) - n_train
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )
    elif args.train and args.val:
        train_ds = LatentsDataset(args.train)
        val_ds = LatentsDataset(args.val)
    else:
        raise ValueError("Provide either --data (single file) or both --train and --val")

    # When using random_split the dataset is a Subset; unwrap for metadata access.
    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
    n_latents = base_ds.latents.shape[1]
    # Compute positive rate over training indices only
    train_labels = (
        base_ds.correctness[list(train_ds.indices)]
        if hasattr(train_ds, "indices")
        else base_ds.correctness
    )
    pos_rate = train_labels.float().mean().item()
    print(f"Training samples  : {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"n_latents         : {n_latents}")
    print(
        f"Positive rate     : {pos_rate:.3f}  (majority baseline: {max(pos_rate, 1 - pos_rate):.3f})"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = StepCorrectnessClassifier(
        input_dim=n_latents,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Acc':>8}")
    print("─" * 30)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        marker = " *" if val_acc > best_acc else ""
        print(f"{epoch:>5}  {loss:>10.4f}  {val_acc:>8.4f}{marker}")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save(out)

    print(f"\nBest val accuracy : {best_acc:.4f}  ({best_acc * 100:.2f}%)")
    print(f"Probe saved to    : {out}")
    print("\nPaper target (SSAE-Qwen GSM8K): 78.58%")


if __name__ == "__main__":
    main()
