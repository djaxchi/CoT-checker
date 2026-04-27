#!/usr/bin/env python3
"""SSAE training on GSM8K-Aug (reconstruction phase).

Trains the context-conditioned encoder + sparse projector + projection_mlp +
decoder to reconstruct individual reasoning steps from a sparse latent h_hat.

Loss: NLL_reconstruct + lambda * ||h_hat||_1
Lambda is adaptively controlled by a DWA controller targeting L1_TARGET active
units on average (paper: τ_spar = 3.0).

Outputs per-optimizer-step logs to <output_dir>/train_log.jsonl:
    {"epoch", "step", "nll", "sparsity", "l1_weight", "lr",
     "n_active_frac", "h_hat_l1_mean"}
Validation rows (written after each epoch):
    {"epoch", "val_nll", "val_sparsity"}

Usage:
    python scripts/train_ssae.py \\
        --data data/gsm8k_385K_train.json \\
        --val-data data/gsm8k_385K_valid.json \\
        --output-dir results/checkpoints/ssae_gsm8k \\
        --device cuda

    # Override any hyperparameter:
    python scripts/train_ssae.py \\
        --data data/gsm8k_385K_train.json \\
        --output-dir results/checkpoints/ssae_gsm8k \\
        --l1-target 10.0 --epochs 30
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
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k_dataset import GSM8KCollateFn, GSM8KStepDataset
from src.saes.ssae import SSAE, TopKActivation


# ---------------------------------------------------------------------------
# DWA controller (eq. 7 in paper)
# ---------------------------------------------------------------------------

class DWAController:
    """Dynamic Weight Adjuster for the L1 sparsity penalty.

    Accumulates per-batch sparsity losses and nudges the L1 weight
    multiplicatively every `update_freq` batches to keep average sparsity
    near `target`. Alpha = 0.01 (1%) per paper.
    """

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

    def step(self, sparsity_loss: float) -> float:
        """Record one batch's sparsity loss and (possibly) update the weight.

        Returns the current L1 weight (pre-update, matching reference behavior
        where the weight read at the start of each batch is the one used).
        """
        self._accumulator += sparsity_loss
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
# Phase 1 loss
# ---------------------------------------------------------------------------

def phase1_nll_loss(
    logits: torch.Tensor,    # (B, T, V)
    input_ids: torch.Tensor, # (B, T)
    loss_mask: torch.Tensor, # (B, T)  1 = step tokens
) -> torch.Tensor:
    """Cross-entropy on step tokens only (context tokens are masked out)."""
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.view(-1, V), input_ids.view(-1), reduction="none")
    ce = ce.view(B, T)
    denom = loss_mask.float().sum().clamp(min=1.0)
    return (ce * loss_mask.float()).sum() / denom


# ---------------------------------------------------------------------------
# Step-region attention masking (paper §3.2, σ = 0.10)
# ---------------------------------------------------------------------------

def apply_step_attention_mask(
    attention_mask: torch.Tensor,  # (B, T)
    sep_pos: torch.Tensor,         # (B,)  first step-token index
    val_len: torch.Tensor,         # (B,)  total non-padded sequence length
    mask_prob: float,
) -> torch.Tensor:
    """Randomly zero out mask_prob fraction of step-token positions in the
    encoder attention mask.  Context tokens are never touched.
    Faithful to the reference: masks attention rather than replacing token ids.
    """
    if mask_prob <= 0.0:
        return attention_mask
    B, T = attention_mask.shape
    device = attention_mask.device
    pos = torch.arange(T, device=device).unsqueeze(0)          # (1, T)
    step_region = (pos >= sep_pos.unsqueeze(1)) & (pos < val_len.unsqueeze(1))
    keep = torch.rand(B, T, device=device) >= mask_prob
    masked = attention_mask.clone()
    masked[step_region & ~keep] = 0
    return masked


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path, model: SSAE, optimizer, step: int, best_val_loss: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "encoder_name": model.encoder.config._name_or_path,
            "decoder_name": model.decoder.config._name_or_path,
            "sparsity_factor": model.sparsity_factor,
            "config": {
                "sparsity_factor": model.sparsity_factor,
                "phase": 1,
                "freeze_encoder": model.freeze_encoder,
            },
        },
        path,
    )
    print(f"  [ckpt] {path}  step={step}  val_nll={best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_phase1(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Tokenizer + model ---
    print(f"Initialising SSAE from {args.model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "<sep>"})
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    activation = TopKActivation(args.topk_k) if args.topk_k > 0 else None

    dtype = None
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16

    model = SSAE(
        tokenizer=tokenizer,
        sparsity_factor=args.sparsity_factor,
        activation=activation,
        phase=1,
        dtype=dtype,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    act_name = f"TopK(k={args.topk_k})" if args.topk_k > 0 else "ReLU+L1"
    print(f"  n_inputs={model.n_inputs}  n_latents={model.n_latents}  "
          f"trainable_params={n_trainable:,}")
    print(f"  activation={act_name}  freeze_encoder={args.freeze_encoder}  dtype={args.dtype}")

    eos_id = tokenizer.eos_token_id
    pad_id = eos_id  # Qwen uses eos as pad
    sep_id = tokenizer.sep_token_id

    # --- Dataset ---
    print(f"Loading train data from {args.data} …")
    full_ds = GSM8KStepDataset(args.data, tokenizer, max_length=args.max_length)

    if args.val_data:
        train_ds = full_ds
        val_ds = GSM8KStepDataset(args.val_data, tokenizer, max_length=args.max_length)
    else:
        n_val = max(1, int(len(full_ds) * args.val_frac))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )

    collate = GSM8KCollateFn(
        eos_token_id=eos_id, pad_token_id=pad_id, sep_token_id=sep_id
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
        num_workers=args.num_workers,
    )
    print(f"  Train: {len(train_ds)} steps  |  Val: {len(val_ds)} steps  "
          f"|  Batches/epoch: {len(train_loader)}")

    # --- Optimizer ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    total_grad_steps = (len(train_loader) // args.grad_accum) * args.epochs

    # --- Controllers ---
    dwa = DWAController(
        target=args.l1_target,
        init_weight=args.l1_weight_init,
        update_freq=args.l1_dwa_interval,
    )

    # --- Output ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "best.pt"

    best_val = float("inf")
    global_step = 0

    use_topk = args.topk_k > 0
    print(f"\nPhase 1 training: {args.epochs} epochs  "
          f"lr={args.lr:.0e}  grad_accum={args.grad_accum}")
    if use_topk:
        print(f"  TopK sparsity: k={args.topk_k}  (L1 penalty disabled)")
    else:
        print(f"  L1 sparsity: target={args.l1_target}  l1_init={args.l1_weight_init:.0e}")

    with open(log_path, "w") as log_f:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            accum = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                hints_sep_ids = batch["hints_sep_ids"].to(device)
                hints_sep_attn = batch["hints_sep_attention_masks"].to(device)
                sep_pos = torch.as_tensor(batch["sep_pos"], device=device)
                val_len = torch.as_tensor(batch["val_len"], device=device)

                # 10% step-token attention masking for robustness (paper §3.2)
                masked_attn = apply_step_attention_mask(
                    attn_mask, sep_pos, val_len, args.mask_prob
                )

                l1_w = 0.0 if use_topk else dwa.current_weight

                latents, loss_sparsity, logits = model(
                    input_ids, masked_attn, hints_sep_ids, hints_sep_attn
                )

                # Align logits with input_ids (slice off sparsity_factor-1 prefix)
                logits = logits[:, args.sparsity_factor - 1:, :]

                loss_nll = phase1_nll_loss(logits, input_ids, loss_mask)
                loss_spa = loss_sparsity / input_ids.shape[0]  # mean over batch
                loss = loss_nll + l1_w * loss_spa

                (loss / args.grad_accum).backward()
                accum += 1

                if not use_topk:
                    dwa.step(loss_spa.item())  # update per batch, matching reference

                if accum == args.grad_accum:
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    lr = get_lr(
                        global_step, args.lr, args.min_lr,
                        args.warmup_steps, total_grad_steps,
                    )
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad()
                    accum = 0
                    global_step += 1

                    # h_hat statistics (no extra forward pass — reuse latents)
                    with torch.no_grad():
                        h = latents.squeeze(1)  # (B, n_latents)
                        n_active_frac = (h.abs() > 1e-3).float().mean().item()
                        h_hat_l1_mean = h.abs().mean().item()

                    log_f.write(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": global_step,
                                "nll": loss_nll.item(),
                                "sparsity": loss_spa.item(),
                                "l1_weight": l1_w,
                                "lr": lr,
                                "n_active_frac": n_active_frac,
                                "h_hat_l1_mean": h_hat_l1_mean,
                            }
                        )
                        + "\n"
                    )
                    log_f.flush()

            # --- Validation ---
            model.eval()
            val_nll_total = 0.0
            val_spa_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attention_mask"].to(device)
                    loss_mask = batch["loss_mask"].to(device)
                    hints_sep_ids = batch["hints_sep_ids"].to(device)
                    hints_sep_attn = batch["hints_sep_attention_masks"].to(device)

                    latents, loss_sparsity, logits = model(
                        input_ids, attn_mask, hints_sep_ids, hints_sep_attn
                    )
                    logits = logits[:, args.sparsity_factor - 1:, :]
                    val_nll_total += phase1_nll_loss(logits, input_ids, loss_mask).item()
                    val_spa_total += (loss_sparsity / input_ids.shape[0]).item()

            val_nll = val_nll_total / max(len(val_loader), 1)
            val_spa = val_spa_total / max(len(val_loader), 1)

            log_f.write(
                json.dumps(
                    {"epoch": epoch + 1, "val_nll": val_nll, "val_sparsity": val_spa}
                )
                + "\n"
            )
            log_f.flush()

            print(
                f"  Epoch {epoch + 1}: val_nll={val_nll:.4f}  "
                f"val_spa={val_spa:.3f}  l1w={dwa.current_weight:.2e}  "
                f"best={best_val:.4f}"
            )

            if val_nll < best_val:
                best_val = val_nll
                save_checkpoint(ckpt_path, model, optimizer, global_step, best_val)

    print(f"\nDone. Best val_nll={best_val:.4f}  →  {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 SSAE training on GSM8K-Aug")
    # Data
    p.add_argument("--data", required=True, help="Path to train JSONL")
    p.add_argument("--val-data", default=None, help="Separate val JSONL (else auto-split)")
    p.add_argument("--output-dir", required=True, help="Dir for checkpoints + train_log.jsonl")
    # Model
    p.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--sparsity-factor", type=int, default=1)
    p.add_argument("--topk-k", type=int, default=0,
                   help="Use TopK activation with exactly K active features. "
                        "0 = disabled (use ReLU + L1 instead).")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze the backbone encoder during phase 1 training.")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help="Model dtype. bfloat16 recommended for H100.")
    # Sparsity
    p.add_argument("--l1-target", type=float, default=3.0)
    p.add_argument("--l1-weight-init", type=float, default=1e-4)
    p.add_argument("--l1-dwa-interval", type=int, default=100)
    # Optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    # Noise
    p.add_argument("--mask-prob", type=float, default=0.10)
    # Data loading
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    # Hardware
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


if __name__ == "__main__":
    train_phase1(parse_args())
