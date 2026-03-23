#!/usr/bin/env python3
"""Step 3 — Evaluate a trained correctness probe on held-out data.

Reports accuracy and compares against the paper's reported numbers.

Usage:
    python scripts/eval_probe.py \\
        --probe  results/probes/correctness_probe.pt \\
        --data   results/probe_data/gsm8k_valid.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probes.classifier import StepCorrectnessClassifier


class LatentsDataset(Dataset):
    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)
        self.latents = torch.from_numpy(data["latents"].astype(np.float32))
        self.correctness = torch.from_numpy(data["correctness"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.correctness[idx]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate step-correctness probe")
    p.add_argument("--probe", required=True, help="Path to saved probe .pt")
    p.add_argument("--data", required=True, help="Path to .npz probe data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, threshold):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(-1)
        preds = (torch.sigmoid(logits) >= threshold).long()
        correct += (preds == y.long()).sum().item()
        total += len(y)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.long().cpu().tolist())
    acc = correct / total if total else 0.0

    # Per-class breakdown
    labels_t = torch.tensor(all_labels)
    preds_t = torch.tensor(all_preds)
    tp = ((preds_t == 1) & (labels_t == 1)).sum().item()
    fp = ((preds_t == 1) & (labels_t == 0)).sum().item()
    fn = ((preds_t == 0) & (labels_t == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return acc, total, precision, recall, f1


def main():
    args = parse_args()
    device = args.device

    model = StepCorrectnessClassifier.load(args.probe, device=device)
    dataset = LatentsDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    pos_rate = dataset.correctness.mean().item()
    majority = max(pos_rate, 1 - pos_rate)

    acc, n, precision, recall, f1 = evaluate(model, loader, device, args.threshold)

    print(f"\n{'─' * 50}")
    print("  Step Correctness Evaluation")
    print(f"{'─' * 50}")
    print(f"  Samples          : {n}")
    print(f"  Positive rate    : {pos_rate:.3f}")
    print("")
    print(f"  Majority baseline: {majority * 100:.2f}%")
    print(f"  Probe accuracy   : {acc * 100:.2f}%")
    print(f"  Improvement      : +{(acc - majority) * 100:.2f}pp")
    print("")
    print(f"  Precision        : {precision:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"  F1               : {f1:.4f}")
    print(f"{'─' * 50}")
    print("\n  Paper target (SSAE-Qwen GSM8K): 78.58%")
    if acc * 100 >= 77.0:
        print("  ✓ Within range of paper result")
    else:
        print("  ✗ Below paper result — check data / training config")
    print()


if __name__ == "__main__":
    main()
