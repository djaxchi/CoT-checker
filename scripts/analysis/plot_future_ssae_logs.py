#!/usr/bin/env python3
"""Plot training diagnostics from a Future-SSAE train_log.jsonl.

Usage:
    python scripts/analysis/plot_future_ssae_logs.py \
        --log results/checkpoints/future_ssae_m1_alpha01_qcurrent/train_log.jsonl \
        --output results/plots/future_ssae_m1_alpha01_qcurrent

Outputs (written to --output dir):
    loss_curves.png
    first_token_stratified.png
    pred_nll_stratified.png
    corr_recon_pred.png
    sparsity_curve.png

Decision rules printed to stdout after plotting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(path: Path) -> tuple[list[dict], list[dict]]:
    train_rows, val_rows = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "val_recon_nll" in row:
                val_rows.append(row)
            else:
                train_rows.append(row)
    return train_rows, val_rows


def _get(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if key in r]


def plot_loss_curves(train: list[dict], out_dir: Path) -> None:
    steps = [r["step"] for r in train]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Loss Curves")

    axes[0, 0].plot(steps, _get(train, "nll_recon"), lw=0.8)
    axes[0, 0].set_title("nll_recon")
    axes[0, 0].set_xlabel("step")

    axes[0, 1].plot(steps, _get(train, "nll_pred"), lw=0.8, color="tab:orange")
    axes[0, 1].set_title("nll_pred")
    axes[0, 1].set_xlabel("step")

    axes[1, 0].plot(steps, _get(train, "loss_total"), lw=0.8, color="tab:red")
    axes[1, 0].set_title("loss_total")
    axes[1, 0].set_xlabel("step")

    axes[1, 1].plot(steps, _get(train, "l1_weight"), lw=0.8, color="tab:purple")
    axes[1, 1].set_title("l1_weight (DWA)")
    axes[1, 1].set_xlabel("step")

    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def plot_first_token_stratified(val: list[dict], out_dir: Path) -> None:
    epochs = [r["epoch"] for r in val]
    idx0 = _get(val, "val_pred_first_token_nll_idx0")
    idx1 = _get(val, "val_pred_first_token_nll_idx1")
    idx2 = _get(val, "val_pred_first_token_nll_idx2plus")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs[:len(idx0)], idx0, marker="o", label="idx=0")
    ax.plot(epochs[:len(idx1)], idx1, marker="o", label="idx=1")
    ax.plot(epochs[:len(idx2)], idx2, marker="o", label="idx=2+")
    ax.set_title("First-token pred NLL by step-index bucket")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "first_token_stratified.png", dpi=150)
    plt.close(fig)


def plot_pred_nll_stratified(val: list[dict], out_dir: Path) -> None:
    epochs = [r["epoch"] for r in val]
    idx0 = _get(val, "val_pred_nll_idx0")
    idx1 = _get(val, "val_pred_nll_idx1")
    idx2 = _get(val, "val_pred_nll_idx2plus")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs[:len(idx0)], idx0, marker="o", label="idx=0")
    ax.plot(epochs[:len(idx1)], idx1, marker="o", label="idx=1")
    ax.plot(epochs[:len(idx2)], idx2, marker="o", label="idx=2+")
    ax.set_title("Full pred NLL by step-index bucket")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pred_nll_stratified.png", dpi=150)
    plt.close(fig)


def plot_corr_recon_pred(val: list[dict], out_dir: Path) -> None:
    epochs = [r["epoch"] for r in val]
    corr = _get(val, "val_corr_recon_pred")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs[:len(corr)], corr, marker="o", color="tab:red")
    ax.axhline(-0.5, ls="--", color="gray", lw=0.8, label="conflict threshold (-0.5)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("val_corr_recon_pred (Pearson r)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("r")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "corr_recon_pred.png", dpi=150)
    plt.close(fig)


def plot_sparsity(train: list[dict], out_dir: Path) -> None:
    steps = [r["step"] for r in train]
    sparsity = _get(train, "sparsity")
    n_active = _get(train, "n_active_frac")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps[:len(sparsity)], sparsity, lw=0.8, color="tab:green")
    axes[0].set_title("sparsity")
    axes[0].set_xlabel("step")

    axes[1].plot(steps[:len(n_active)], n_active, lw=0.8, color="tab:brown")
    axes[1].set_title("n_active_frac")
    axes[1].set_xlabel("step")

    fig.suptitle("Sparsity Diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "sparsity_curve.png", dpi=150)
    plt.close(fig)


def print_decision_rules(val: list[dict]) -> None:
    if not val:
        print("No validation rows — cannot evaluate decision rules.")
        return

    def trend_down(vals: list[float]) -> bool:
        if len(vals) < 2:
            return False
        return vals[-1] < vals[0]

    idx0 = _get(val, "val_pred_first_token_nll_idx0")
    idx1 = _get(val, "val_pred_first_token_nll_idx1")
    idx2 = _get(val, "val_pred_first_token_nll_idx2plus")
    corr = _get(val, "val_corr_recon_pred")

    d0 = trend_down(idx0)
    d1 = trend_down(idx1)
    d2 = trend_down(idx2)

    print("\n=== Decision Rules ===")
    if d0 and d1 and d2:
        print("GOOD: pred_first_token_nll_idx0, idx1, idx2plus decrease")
    elif d0 and not (d1 and d2):
        print("PARTIAL: idx0 decreases but idx1/idx2plus flat")
    else:
        print("BAD: all pred_first_token_nll curves flat")

    if corr and corr[-1] < -0.5:
        print("CONFLICT: val_corr_recon_pred < -0.5")

    print(f"\nFinal val values (last epoch):")
    last = val[-1]
    for k in (
        "val_pred_first_token_nll_idx0",
        "val_pred_first_token_nll_idx1",
        "val_pred_first_token_nll_idx2plus",
        "val_corr_recon_pred",
        "val_n_idx0",
        "val_n_idx1",
        "val_n_idx2plus",
        "val_total",
    ):
        if k in last:
            print(f"  {k}: {last[k]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="Path to train_log.jsonl")
    p.add_argument("--output", required=True, help="Output directory for plots")
    args = p.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.output)

    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    train, val = load_log(log_path)
    print(f"Loaded {len(train)} train rows, {len(val)} val rows from {log_path}")

    if not train and not val:
        print("ERROR: log is empty.", file=sys.stderr)
        sys.exit(1)

    if train:
        plot_loss_curves(train, out_dir)
        plot_sparsity(train, out_dir)
        print(f"  loss_curves.png")
        print(f"  sparsity_curve.png")

    if val:
        plot_first_token_stratified(val, out_dir)
        plot_pred_nll_stratified(val, out_dir)
        plot_corr_recon_pred(val, out_dir)
        print(f"  first_token_stratified.png")
        print(f"  pred_nll_stratified.png")
        print(f"  corr_recon_pred.png")
        print_decision_rules(val)
    else:
        print("WARNING: no validation rows found — skipping stratified plots and decision rules.")


if __name__ == "__main__":
    main()
