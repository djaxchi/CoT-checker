#!/usr/bin/env python3
"""Train the Predictive Transition Bottleneck (PTB).

Objective:
    h_k -> z_k -> delta_h_hat_k ~= h_{k+1} - h_k

Loss (varies by --dwa-mode):
    L = MSE(delta_h_hat_k, delta_h_k) + lambda * sparsity_term

DWA modes:
    dwa_original        : original behavior, L1 target 3.0 (for reproduction only)
    dwa_calibrated_mean_l1 : L1 target set from calibration pass on actual z magnitudes
    dwa_active_fraction : control sparsity via active-dim fraction instead of raw L1
    fixed_lambda        : no DWA, fixed L1 weight (--l1-weight-init)
    no_l1               : no sparsity penalty (transition-only baseline)
    topk                : exact k-sparse via TopK activation, no L1 penalty

Usage:
    python scripts/train_predictive_transition_bottleneck.py \\
        --train-data  transition_train_positive.npz \\
        --val-data    transition_val_positive.npz \\
        --output-dir  results/checkpoints/ptb_no_l1 \\
        --dwa-mode    no_l1 \\
        --device      cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.saes.ptb import PredictiveTransitionBottleneck
from src.data.transition_dataset import TransitionDataset


# ---------------------------------------------------------------------------
# DWA controllers
# ---------------------------------------------------------------------------

class MeanL1DWA:
    """Dynamic weight adapter targeting mean L1 of z activations."""

    def __init__(
        self,
        target: float,
        init_weight: float = 1e-4,
        update_freq: int = 100,
        alpha: float = 0.01,
        min_w: float = 1e-6,
        max_w: float = 0.1,
    ) -> None:
        self.target = target
        self.weight = init_weight
        self.update_freq = update_freq
        self.alpha = alpha
        self.min_w = min_w
        self.max_w = max_w
        self._acc = 0.0
        self._steps = 0
        self.n_updates = 0
        self.n_min_clips = 0
        self.n_max_clips = 0

    def step(self, val: float) -> float:
        self._acc += val
        self._steps += 1
        if self._steps >= self.update_freq:
            avg = self._acc / self._steps
            direction = 1 if avg > self.target else -1
            new_w = self.weight * (1.0 + direction * self.alpha)
            if new_w <= self.min_w:
                self.n_min_clips += 1
            if new_w >= self.max_w:
                self.n_max_clips += 1
            self.weight = max(self.min_w, min(self.max_w, new_w))
            self._acc = 0.0
            self._steps = 0
            self.n_updates += 1
        return self.weight

    @property
    def current_weight(self) -> float:
        return self.weight

    def saturation_report(self) -> dict:
        total_updates = max(self.n_updates, 1)
        return {
            "n_updates":     self.n_updates,
            "n_min_clips":   self.n_min_clips,
            "n_max_clips":   self.n_max_clips,
            "min_clip_frac": self.n_min_clips / total_updates,
            "max_clip_frac": self.n_max_clips / total_updates,
            "final_weight":  self.weight,
        }


class ActiveFractionDWA:
    """DWA targeting a fraction of active dimensions per sample (per threshold)."""

    def __init__(
        self,
        target_frac: float,     # e.g. 0.25 = 25% of dims active
        eps: float = 1e-3,
        init_weight: float = 1e-4,
        update_freq: int = 100,
        alpha: float = 0.01,
        min_w: float = 1e-6,
        max_w: float = 0.1,
    ) -> None:
        self.target_frac = target_frac
        self.eps = eps
        self.weight = init_weight
        self.update_freq = update_freq
        self.alpha = alpha
        self.min_w = min_w
        self.max_w = max_w
        self._acc = 0.0
        self._steps = 0
        self.n_updates = 0
        self.n_min_clips = 0
        self.n_max_clips = 0

    def step(self, z: torch.Tensor) -> float:
        frac = (z.abs() > self.eps).float().mean().item()
        self._acc += frac
        self._steps += 1
        if self._steps >= self.update_freq:
            avg = self._acc / self._steps
            direction = 1 if avg > self.target_frac else -1
            new_w = self.weight * (1.0 + direction * self.alpha)
            if new_w <= self.min_w:
                self.n_min_clips += 1
            if new_w >= self.max_w:
                self.n_max_clips += 1
            self.weight = max(self.min_w, min(self.max_w, new_w))
            self._acc = 0.0
            self._steps = 0
            self.n_updates += 1
        return self.weight

    @property
    def current_weight(self) -> float:
        return self.weight

    def saturation_report(self) -> dict:
        total = max(self.n_updates, 1)
        return {
            "n_updates":     self.n_updates,
            "n_min_clips":   self.n_min_clips,
            "n_max_clips":   self.n_max_clips,
            "min_clip_frac": self.n_min_clips / total,
            "max_clip_frac": self.n_max_clips / total,
            "final_weight":  self.weight,
        }


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, lr: float, min_lr: float, warmup: int, decay_iters: int) -> float:
    if step < warmup:
        return lr * (step + 1) / (warmup + 1)
    if step > decay_iters:
        return min_lr
    t = (step - warmup) / max(decay_iters - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Calibration pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_z_statistics(
    model: PredictiveTransitionBottleneck,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 50,
) -> dict:
    """Run encoder on first n_batches to measure z activation statistics.

    Returns a dict with z_mean, z_median, z_max, active fractions at four
    eps thresholds, mean active dims, and a recommended L1 target.
    """
    model.eval()
    z_means, z_medians, z_maxes = [], [], []
    active_frac_6, active_frac_5, active_frac_4, active_frac_3 = [], [], [], []
    mean_active_dims_3 = []
    delta_h_norms = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        h_k     = batch["h_k"].to(device)
        delta_h = batch["delta_h"].to(device)
        z, _    = model(h_k)

        z_means.append(z.abs().mean().item())
        z_medians.append(z.abs().median().item())
        z_maxes.append(z.abs().max().item())
        active_frac_6.append((z.abs() > 1e-6).float().mean().item())
        active_frac_5.append((z.abs() > 1e-5).float().mean().item())
        active_frac_4.append((z.abs() > 1e-4).float().mean().item())
        active_frac_3.append((z.abs() > 1e-3).float().mean().item())
        mean_active_dims_3.append((z.abs() > 1e-3).float().sum(1).mean().item())
        delta_h_norms.append(delta_h.norm(dim=1).mean().item())

    n_done = min(n_batches, i + 1)

    def m(lst: list) -> float:
        return float(np.mean(lst))

    result = {
        "n_batches":              n_done,
        "z_mean":                 m(z_means),
        "z_median":               m(z_medians),
        "z_max":                  m(z_maxes),
        "active_frac_1e-6":       m(active_frac_6),
        "active_frac_1e-5":       m(active_frac_5),
        "active_frac_1e-4":       m(active_frac_4),
        "active_frac_1e-3":       m(active_frac_3),
        "mean_active_dims_1e-3":  m(mean_active_dims_3),
        "delta_h_norm_mean":      m(delta_h_norms),
        # Recommended L1 target: 10% of empirical z_mean (well below initial scale)
        "recommended_l1_target":  m(z_means) * 0.1,
    }
    return result


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_step_metrics(
    z: torch.Tensor,
    delta_h: torch.Tensor,
    delta_hat: torch.Tensor,
    enc_weight: torch.nn.Parameter,
    dec_weight: torch.nn.Parameter,
) -> dict:
    """Compute comprehensive per-batch metrics."""
    mse  = F.mse_loss(delta_hat, delta_h).item()
    l1   = z.abs().mean().item()
    cos  = F.cosine_similarity(delta_hat, delta_h, dim=-1).mean().item()
    # R2: fraction of delta_h variance explained
    ss_res = F.mse_loss(delta_hat, delta_h, reduction="sum").item()
    ss_tot = ((delta_h - delta_h.mean(0, keepdim=True)) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

    return {
        "mse_loss":            mse,
        "l1_z":                l1,
        "cosine_sim":          cos,
        "r2":                  r2,
        "z_mean":              z.abs().mean().item(),
        "z_median":            float(z.abs().median().item()),
        "z_max":               z.abs().max().item(),
        "active_frac_1e-6":    (z.abs() > 1e-6).float().mean().item(),
        "active_frac_1e-5":    (z.abs() > 1e-5).float().mean().item(),
        "active_frac_1e-4":    (z.abs() > 1e-4).float().mean().item(),
        "active_frac_1e-3":    (z.abs() > 1e-3).float().mean().item(),
        "mean_active_dims":    (z.abs() > 1e-3).float().sum(1).mean().item(),
        "enc_weight_norm":     enc_weight.norm().item(),
        "dec_weight_norm":     dec_weight.norm().item(),
        "delta_h_norm":        delta_h.norm(dim=1).mean().item(),
        "delta_h_hat_norm":    delta_hat.norm(dim=1).mean().item(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading train data from {args.train_data} ...")
    full_ds = TransitionDataset(args.train_data)

    if args.val_data:
        train_ds = full_ds
        val_ds   = TransitionDataset(args.val_data)
    else:
        n_val   = max(1, int(len(full_ds) * args.val_frac))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Batches/epoch: {len(train_loader)}")

    k = args.l1_k if args.dwa_mode == "topk" else None
    model = PredictiveTransitionBottleneck(d=args.d, k=k).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    n_gpus = torch.cuda.device_count() if args.device == "cuda" else 0
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"  Model: d={args.d}  k={k}  params={n_params:,}  DataParallel x{n_gpus}")
    else:
        print(f"  Model: d={args.d}  k={k}  params={n_params:,}")

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Calibration pass (always run; determines L1 target for calibrated modes)
    # ------------------------------------------------------------------
    print("\n=== Calibration pass ===")
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    calib = calibrate_z_statistics(core, train_loader, device, n_batches=args.calib_batches)
    calib_path = out_dir / "calibration.json"
    calib_path.write_text(json.dumps(calib, indent=2))
    print(f"  z_mean={calib['z_mean']:.4f}  z_median={calib['z_median']:.4f}")
    print(f"  active@1e-3={calib['active_frac_1e-3']:.3f}  mean_active_dims={calib['mean_active_dims_1e-3']:.1f}")
    print(f"  recommended_l1_target={calib['recommended_l1_target']:.4f}")
    print(f"  Calibration -> {calib_path}")

    # ------------------------------------------------------------------
    # Set up DWA / sparsity controller based on mode
    # ------------------------------------------------------------------
    dwa_mode = args.dwa_mode
    l1_target_effective = args.l1_target
    dwa = None

    if dwa_mode == "dwa_original":
        l1_target_effective = args.l1_target  # 3.0 default, inherited from SSAE
        dwa = MeanL1DWA(
            target=l1_target_effective,
            init_weight=args.l1_weight_init,
            update_freq=args.l1_dwa_interval,
        )
        print(f"\nDWA mode: dwa_original  target={l1_target_effective}")

    elif dwa_mode == "dwa_calibrated_mean_l1":
        # Use empirical z_mean from calibration as target
        l1_target_effective = calib["recommended_l1_target"]
        dwa = MeanL1DWA(
            target=l1_target_effective,
            init_weight=args.l1_weight_init,
            update_freq=args.l1_dwa_interval,
        )
        print(f"\nDWA mode: dwa_calibrated_mean_l1  target={l1_target_effective:.4f}  "
              f"(empirical z_mean={calib['z_mean']:.4f} * 0.1)")

    elif dwa_mode == "dwa_active_fraction":
        target_frac = args.l1_target_active_frac
        dwa = ActiveFractionDWA(
            target_frac=target_frac,
            eps=1e-3,
            init_weight=args.l1_weight_init,
            update_freq=args.l1_dwa_interval,
        )
        print(f"\nDWA mode: dwa_active_fraction  target_frac={target_frac}")

    elif dwa_mode == "fixed_lambda":
        print(f"\nDWA mode: fixed_lambda  lambda={args.l1_weight_init}")

    elif dwa_mode == "no_l1":
        print("\nDWA mode: no_l1  (transition-only, no sparsity penalty)")

    elif dwa_mode == "topk":
        print(f"\nDWA mode: topk  k={args.l1_k}  (exact sparsity, no L1 penalty)")

    else:
        raise ValueError(f"Unknown --dwa-mode: {dwa_mode}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs

    log_path  = out_dir / "train_log.jsonl"
    csv_path  = out_dir / "train_metrics.csv"
    ckpt_path = out_dir / "best.pt"

    # CSV header
    csv_fields = [
        "epoch", "step", "phase",
        "mse_loss", "l1_z", "l1_weight", "total_loss",
        "cosine_sim", "r2",
        "z_mean", "z_median", "z_max",
        "active_frac_1e-6", "active_frac_1e-5", "active_frac_1e-4", "active_frac_1e-3",
        "mean_active_dims",
        "enc_weight_norm", "dec_weight_norm",
        "delta_h_norm", "delta_h_hat_norm",
        "lr",
    ]

    best_val = float("inf")
    global_step = 0

    print(f"\nPTB training: {args.epochs} epochs  lr={args.lr:.0e}  "
          f"dwa_mode={dwa_mode}  batch={args.batch_size}")
    print(f"  Logs: {log_path}")

    with open(log_path, "w") as log_f, open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            accum = 0

            for batch in train_loader:
                h_k     = batch["h_k"].to(device)
                delta_h = batch["delta_h"].to(device)

                # Current lambda
                if dwa_mode in ("dwa_original", "dwa_calibrated_mean_l1"):
                    l1_w = dwa.current_weight
                elif dwa_mode == "dwa_active_fraction":
                    l1_w = dwa.current_weight
                elif dwa_mode == "fixed_lambda":
                    l1_w = args.l1_weight_init
                else:
                    l1_w = 0.0

                z_k, delta_hat = model(h_k)
                mse = F.mse_loss(delta_hat, delta_h)

                if dwa_mode == "no_l1" or dwa_mode == "topk":
                    loss = mse
                elif dwa_mode == "dwa_active_fraction":
                    l1   = z_k.abs().mean()
                    loss = mse + l1_w * l1
                else:
                    l1   = z_k.abs().mean()
                    loss = mse + l1_w * l1

                (loss / args.grad_accum).backward()
                accum += 1

                # DWA step
                if dwa_mode in ("dwa_original", "dwa_calibrated_mean_l1", "fixed_lambda"):
                    if dwa is not None:
                        dwa.step(z_k.abs().mean().item())
                elif dwa_mode == "dwa_active_fraction":
                    with torch.no_grad():
                        dwa.step(z_k)

                if accum == args.grad_accum:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    lr = get_lr(global_step, args.lr, args.min_lr, args.warmup_steps, total_steps)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad()
                    accum = 0
                    global_step += 1

                    with torch.no_grad():
                        c = model.module if isinstance(model, torch.nn.DataParallel) else model
                        metrics = compute_step_metrics(z_k, delta_h, delta_hat,
                                                       c.enc_weight, c.dec.weight)

                    row = {
                        "epoch":    epoch + 1,
                        "step":     global_step,
                        "phase":    "train",
                        "l1_weight": l1_w,
                        "total_loss": loss.item(),
                        "lr":       lr,
                        **metrics,
                    }
                    log_f.write(json.dumps(row) + "\n")
                    log_f.flush()
                    writer.writerow(row)
                    csv_f.flush()

            # ---- Validation ----
            model.eval()
            val_totals: dict[str, float] = {k: 0.0 for k in [
                "mse_loss", "l1_z", "cosine_sim", "r2",
                "z_mean", "z_median", "z_max",
                "active_frac_1e-6", "active_frac_1e-5", "active_frac_1e-4", "active_frac_1e-3",
                "mean_active_dims", "delta_h_norm", "delta_h_hat_norm",
            ]}
            n_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    h_k     = batch["h_k"].to(device)
                    delta_h = batch["delta_h"].to(device)
                    c = model.module if isinstance(model, torch.nn.DataParallel) else model
                    z_k, delta_hat = c(h_k)
                    metrics = compute_step_metrics(z_k, delta_h, delta_hat,
                                                   c.enc_weight, c.dec.weight)
                    for k2 in val_totals:
                        val_totals[k2] += metrics[k2]
                    n_batches += 1

            n_batches = max(n_batches, 1)
            val_avgs = {k2: v / n_batches for k2, v in val_totals.items()}

            # val total = mse + lambda * l1 (same signal for checkpoint selection)
            effective_lambda = l1_w if dwa_mode not in ("no_l1", "topk") else 0.0
            val_total = val_avgs["mse_loss"] + effective_lambda * val_avgs["l1_z"]

            val_row = {
                "epoch": epoch + 1, "step": global_step, "phase": "val",
                "l1_weight": effective_lambda,
                "total_loss": val_total,
                "lr": lr,
                **val_avgs,
            }
            log_f.write(json.dumps(val_row) + "\n")
            log_f.flush()
            writer.writerow(val_row)
            csv_f.flush()

            print(
                f"  Epoch {epoch+1:3d}: "
                f"val_mse={val_avgs['mse_loss']:.5f}  "
                f"val_cos={val_avgs['cosine_sim']:.4f}  "
                f"val_r2={val_avgs['r2']:.4f}  "
                f"val_z_mean={val_avgs['z_mean']:.4f}  "
                f"act@1e-3={val_avgs['active_frac_1e-3']:.3f}  "
                f"mean_active_dims={val_avgs['mean_active_dims']:.1f}  "
                f"l1w={effective_lambda:.2e}  "
                f"best={best_val:.5f}"
            )

            if val_total < best_val:
                best_val = val_total
                c = model.module if isinstance(model, torch.nn.DataParallel) else model
                c.save(ckpt_path, step=global_step, best_val_loss=best_val,
                       extra={"dwa_mode": dwa_mode, "calib": calib})
                print(f"  [ckpt] {ckpt_path}  step={global_step}  val_total={best_val:.5f}")

    # ------------------------------------------------------------------
    # Post-training validity report
    # ------------------------------------------------------------------
    print("\n=== Post-training validity report ===")
    validity: dict = {
        "dwa_mode":          dwa_mode,
        "best_val_loss":     best_val,
        "calibration":       calib,
        "flags":             [],
    }

    # Final z statistics from last val epoch
    validity["final_val_mse"]          = val_avgs["mse_loss"]
    validity["final_val_r2"]           = val_avgs["r2"]
    validity["final_val_cosine_sim"]   = val_avgs["cosine_sim"]
    validity["final_val_z_mean"]       = val_avgs["z_mean"]
    validity["final_val_active_frac"]  = val_avgs["active_frac_1e-3"]
    validity["final_val_mean_active_dims"] = val_avgs["mean_active_dims"]

    # DWA saturation check
    if dwa is not None and hasattr(dwa, "saturation_report"):
        sat = dwa.saturation_report()
        validity["dwa_saturation"] = sat
        if sat["min_clip_frac"] > 0.5:
            msg = (f"INVALID: DWA hit min_weight in {sat['min_clip_frac']*100:.0f}% of updates "
                   f"-- L1 penalty too high, sparsity uncontrolled from below")
            validity["flags"].append(msg)
            print(f"  FLAG: {msg}")
        if sat["max_clip_frac"] > 0.5:
            msg = (f"INVALID: DWA hit max_weight in {sat['max_clip_frac']*100:.0f}% of updates "
                   f"-- L1 penalty too low, sparsity uncontrolled from above")
            validity["flags"].append(msg)
            print(f"  FLAG: {msg}")

    # Transition prediction quality
    if val_avgs["cosine_sim"] < 0.05:
        msg = ("INVALID: val_cosine_sim < 0.05 -- PTB has not learned to predict transitions")
        validity["flags"].append(msg)
        print(f"  FLAG: {msg}")

    if val_avgs["r2"] < 0.0:
        msg = f"WARNING: val_R2 = {val_avgs['r2']:.4f} < 0 -- worse than predicting the mean"
        validity["flags"].append(msg)
        print(f"  FLAG: {msg}")

    # Sparsity sanity
    if dwa_mode not in ("no_l1", "topk") and val_avgs["active_frac_1e-3"] > 0.9:
        msg = ("WARNING: active_frac@1e-3 > 0.9 -- almost no sparsity achieved; "
               "L1 penalty may be ineffective")
        validity["flags"].append(msg)
        print(f"  FLAG: {msg}")

    if not validity["flags"]:
        validity["flags"].append("OK: no critical flags")
        print("  All checks passed.")

    report_path = out_dir / "validity_report.json"
    report_path.write_text(json.dumps(validity, indent=2))
    print(f"  Validity report -> {report_path}")
    print(f"\nDone. Best val_total={best_val:.5f}  ->  {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Predictive Transition Bottleneck (PTB)")

    # Data
    p.add_argument("--train-data",      required=True,  help="transition_train_positive.npz")
    p.add_argument("--val-data",        default=None,   help="Separate val .npz (else auto-split)")
    p.add_argument("--output-dir",      required=True,  help="Dir for best.pt + logs")

    # Architecture
    p.add_argument("--d",               type=int,   default=896)

    # DWA mode
    p.add_argument("--dwa-mode",        default="dwa_original",
                   choices=["dwa_original", "dwa_calibrated_mean_l1",
                             "dwa_active_fraction", "fixed_lambda", "no_l1", "topk"],
                   help="Sparsity control strategy")
    p.add_argument("--l1-target",       type=float, default=3.0,
                   help="Mean-L1 target for dwa_original mode (inherited from SSAE)")
    p.add_argument("--l1-target-active-frac", type=float, default=0.25,
                   help="Target active-dim fraction for dwa_active_fraction mode")
    p.add_argument("--l1-weight-init",  type=float, default=1e-4,
                   help="Initial lambda for DWA modes, or fixed lambda for fixed_lambda")
    p.add_argument("--l1-dwa-interval", type=int,   default=100)
    p.add_argument("--l1-k",            type=int,   default=50,
                   help="Number of active dims for topk mode")

    # Training
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch-size",      type=int,   default=2048)
    p.add_argument("--grad-accum",      type=int,   default=1)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--min-lr",          type=float, default=1e-4)
    p.add_argument("--warmup-steps",    type=int,   default=200)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--val-frac",        type=float, default=0.05)
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--device",          default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed",            type=int,   default=1337)

    # Calibration
    p.add_argument("--calib-batches",   type=int,   default=50,
                   help="Number of batches for calibration pass")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
