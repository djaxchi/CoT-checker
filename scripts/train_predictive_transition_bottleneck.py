#!/usr/bin/env python3
"""Train the Predictive Transition Bottleneck (PTB, C=1).

Objective:
    h_k → z_k → Δh_hat_k ≈ h_{k+1} - h_k

Loss:
    L = MSE(Δh_hat_k, Δh_k) + λ * ||z_k||_1

λ is adapted automatically by a DWA controller targeting a mean L1 sparsity
of --l1-target (same as SSAE training, same DWA implementation).

C=1 is intentional. Latent dim = hidden dim = 896. This experiment isolates
the effect of the training objective (reconstruction vs. transition prediction)
while holding dimensionality constant.

Usage:
    python scripts/train_predictive_transition_bottleneck.py \\
        --train-data $SCRATCH/cot-checker/probe_data/transition_train_positive.npz \\
        --val-data   $SCRATCH/cot-checker/probe_data/transition_val_positive.npz \\
        --output-dir $STORE/results/checkpoints/ptb_c1 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.saes.ptb import PredictiveTransitionBottleneck
from src.data.transition_dataset import TransitionDataset


# ---------------------------------------------------------------------------
# DWA controller (same implementation as train_ssae.py / train_future_ssae_gsm8k.py)
# ---------------------------------------------------------------------------

class DWAController:
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
        self._accumulator = 0.0
        self._steps = 0

    def step(self, sparsity_val: float) -> float:
        self._accumulator += sparsity_val
        self._steps += 1
        if self._steps >= self.update_freq:
            avg = self._accumulator / self._steps
            direction = 1 if avg > self.target else -1
            self.weight = self.weight * (1.0 + direction * self.alpha)
            self.weight = max(self.min_w, min(self.max_w, self.weight))
            self._accumulator = 0.0
            self._steps = 0
        return self.weight

    @property
    def current_weight(self) -> float:
        return self.weight


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

    model = PredictiveTransitionBottleneck(d=args.d).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_gpus = torch.cuda.device_count() if args.device == "cuda" else 0
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"  Model: d={args.d}  params={n_params:,}  DataParallel x{n_gpus}")
    else:
        print(f"  Model: d={args.d}  params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs

    dwa = DWAController(
        target=args.l1_target,
        init_weight=args.l1_weight_init,
        update_freq=args.l1_dwa_interval,
    )

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path  = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "best.pt"

    best_val = float("inf")
    global_step = 0

    print(f"\nPTB training: {args.epochs} epochs  lr={args.lr:.0e}  "
          f"l1_target={args.l1_target}  batch={args.batch_size}")

    with open(log_path, "w") as log_f:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            accum = 0

            for batch in train_loader:
                h_k     = batch["h_k"].to(device)
                delta_h = batch["delta_h"].to(device)

                l1_w = dwa.current_weight

                z_k, delta_hat = model(h_k)

                mse  = F.mse_loss(delta_hat, delta_h)
                l1   = z_k.abs().mean()
                loss = mse + l1_w * l1

                (loss / args.grad_accum).backward()
                accum += 1

                dwa.step(l1.item())

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
                        cos_sim = F.cosine_similarity(delta_hat, delta_h, dim=-1).mean().item()
                        n_active = (z_k.abs() > 1e-3).float().mean().item()

                    log_f.write(json.dumps({
                        "epoch":       epoch + 1,
                        "step":        global_step,
                        "mse_loss":    mse.item(),
                        "cosine_sim":  cos_sim,
                        "l1_sparsity": l1.item(),
                        "loss_total":  loss.item(),
                        "l1_weight":   l1_w,
                        "lr":          lr,
                        "n_active_frac": n_active,
                        "z_l1_mean":   l1.item(),
                    }) + "\n")
                    log_f.flush()

            # ---- Validation ----
            model.eval()
            val_mse_total = val_cos_total = val_l1_total = 0.0
            n_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    h_k     = batch["h_k"].to(device)
                    delta_h = batch["delta_h"].to(device)
                    z_k, delta_hat = model(h_k)
                    val_mse_total += F.mse_loss(delta_hat, delta_h).item()
                    val_cos_total += F.cosine_similarity(delta_hat, delta_h, dim=-1).mean().item()
                    val_l1_total  += z_k.abs().mean().item()
                    n_batches += 1

            n_batches = max(n_batches, 1)
            val_mse = val_mse_total / n_batches
            val_cos = val_cos_total / n_batches
            val_l1  = val_l1_total  / n_batches
            val_total = val_mse + dwa.current_weight * val_l1

            log_f.write(json.dumps({
                "epoch":       epoch + 1,
                "val_mse":     val_mse,
                "val_cosine":  val_cos,
                "val_l1":      val_l1,
                "val_total":   val_total,
            }) + "\n")
            log_f.flush()

            print(f"  Epoch {epoch+1}: val_mse={val_mse:.5f}  val_cos={val_cos:.4f}  "
                  f"val_l1={val_l1:.3f}  val_total={val_total:.5f}  "
                  f"l1w={dwa.current_weight:.2e}  best={best_val:.5f}")

            if val_total < best_val:
                best_val = val_total
                core = model.module if isinstance(model, torch.nn.DataParallel) else model
                core.save(ckpt_path, step=global_step, best_val_loss=best_val)
                print(f"  [ckpt] {ckpt_path}  step={global_step}  val_total={best_val:.5f}")

    print(f"\nDone. Best val_total={best_val:.5f}  →  {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Predictive Transition Bottleneck (PTB, C=1)")
    p.add_argument("--train-data",      required=True,  help="transition_train_positive.npz")
    p.add_argument("--val-data",        default=None,   help="Separate val .npz (else auto-split)")
    p.add_argument("--output-dir",      required=True,  help="Dir for best.pt + train_log.jsonl")
    p.add_argument("--d",               type=int,   default=896)
    p.add_argument("--l1-target",       type=float, default=3.0)
    p.add_argument("--l1-weight-init",  type=float, default=1e-4)
    p.add_argument("--l1-dwa-interval", type=int,   default=100)
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
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
