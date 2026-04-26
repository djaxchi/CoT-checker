#!/usr/bin/env python3
"""Future-SSAE training on GSM8K-Aug.

Extends Phase 1 SSAE with a next-step prediction auxiliary loss:

    L = L_recon + λ * ||h_hat||_1 + α * (L_pred / ema_pred_nll)

The same sparse latent h_hat_k is reused for both branches — the encoder
runs once per batch item.

Reconstruction branch (identical to Phase 1):
    encoder input : C_k + SEP + s_k
    decoder input : C_k + h_hat_k  (teacher forcing)
    target        : s_k

Prediction branch (new):
    encoder input : same (shared h_hat_k, no second forward)
    decoder input : q + s_k + h_hat_k  (or wider context per --pred-context-mode)
    target        : s_{k+1}

Stability diagnostics logged every optimizer step:
    pred_first_token_nll  — NLL at the first token of s_{k+1} only (no s_{k+1}
                            context in the decoder at that position, so gradient
                            flows entirely through h_hat_k)
    pred_nll_scale        — EMA of nll_pred used to normalise the alpha term;
                            tracks when recon and pred are at the same scale

Validation diagnostics (per epoch):
    val_corr_recon_pred   — Pearson r between per-batch nll_recon and nll_pred;
                            strong anti-correlation signals recon/pred competition

Per-step logs → <output_dir>/train_log.jsonl:
    {"epoch", "step", "nll_recon", "nll_pred", "pred_first_token_nll",
     "sparsity", "loss_total", "l1_weight", "alpha_pred", "pred_nll_scale",
     "lr", "n_active_frac", "h_hat_l1_mean"}

Validation rows:
    {"epoch", "val_recon_nll", "val_pred_nll", "val_sparsity", "val_total",
     "val_corr_recon_pred",
     "val_pred_nll_idx0", "val_pred_nll_idx1", "val_pred_nll_idx2plus",
     "val_pred_first_token_nll_idx0", "val_pred_first_token_nll_idx1",
     "val_pred_first_token_nll_idx2plus",
     "val_n_idx0", "val_n_idx1", "val_n_idx2plus"}

The stratified buckets answer "does the prediction branch help beyond the
dominant idx=0 first-step transition?".  GSM8K-Aug audit shows ~52% of pairs
are idx=0, so a model that only improves on idx=0 is learning shallow.

Usage:
    python scripts/train_future_ssae_gsm8k.py \\
        --data data/gsm8k_385K_train.json \\
        --val-data data/gsm8k_385K_valid.json \\
        --output-dir results/checkpoints/future_ssae_gsm8k \\
        --alpha-pred 0.1 \\
        --device cuda

    # Recommended alpha sweep (start here):
    for alpha in 0.03 0.1 0.3 1.0; do
        python scripts/train_future_ssae_gsm8k.py \\
            --data data/gsm8k_385K_train.json \\
            --output-dir results/checkpoints/future_ssae_alpha${alpha} \\
            --alpha-pred ${alpha}
    done

    # Recommended context-mode ablation (q_current first, then q_prev1_current):
    python scripts/train_future_ssae_gsm8k.py \\
        --data ... --output-dir ... --pred-context-mode q_prev1_current
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

from src.data.gsm8k_dataset import GSM8KFutureCollateFn, GSM8KFutureStepDataset
from src.saes.ssae import SSAE


# ---------------------------------------------------------------------------
# DWA controller (identical to Phase 1)
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

    def step(self, sparsity_loss: float) -> float:
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
# Prediction NLL scale tracker (EMA)
# ---------------------------------------------------------------------------

class PredNLLScaler:
    """EMA tracker for nll_pred used to keep the alpha_pred term at the same
    scale as nll_recon regardless of task difficulty.

    Follows the same read-before-write contract as DWAController: call
    current_scale to get the normalisation factor for the current batch, then
    call step() after computing nll_pred to update the EMA.

    The scale is initialised from the first observed nll_pred value so early
    batches don't use a stale default.
    """

    def __init__(self, decay: float = 0.99, min_scale: float = 1e-3) -> None:
        self.decay = decay
        self.min_scale = min_scale
        self._scale: float | None = None  # None until first observation

    def step(self, nll_pred: float) -> None:
        if self._scale is None:
            self._scale = nll_pred
        else:
            self._scale = self.decay * self._scale + (1.0 - self.decay) * nll_pred

    @property
    def current_scale(self) -> float:
        return max(self._scale if self._scale is not None else 1.0, self.min_scale)


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
# Shared NLL loss (step-region mask only)
# ---------------------------------------------------------------------------

def masked_nll_loss(
    logits: torch.Tensor,    # (B, T, V)
    input_ids: torch.Tensor, # (B, T)
    loss_mask: torch.Tensor, # (B, T)  1 = target tokens
) -> torch.Tensor:
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.view(-1, V), input_ids.view(-1), reduction="none")
    ce = ce.view(B, T)
    denom = loss_mask.float().sum().clamp(min=1.0)
    return (ce * loss_mask.float()).sum() / denom


def per_item_masked_nll(
    logits: torch.Tensor,    # (B, T, V)
    input_ids: torch.Tensor, # (B, T)
    loss_mask: torch.Tensor, # (B, T)
) -> torch.Tensor:
    """Per-item NLL: cross-entropy averaged over each item's masked tokens.

    Returns a (B,) tensor.  Used to stratify validation losses by step index.
    """
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.view(-1, V), input_ids.view(-1), reduction="none").view(B, T)
    denom = loss_mask.float().sum(dim=1).clamp(min=1.0)
    return (ce * loss_mask.float()).sum(dim=1) / denom


# ---------------------------------------------------------------------------
# Step-region attention masking (paper §3.2, σ = 0.10)
# ---------------------------------------------------------------------------

def apply_step_attention_mask(
    attention_mask: torch.Tensor,
    sep_pos: torch.Tensor,
    val_len: torch.Tensor,
    mask_prob: float,
) -> torch.Tensor:
    if mask_prob <= 0.0:
        return attention_mask
    B, T = attention_mask.shape
    device = attention_mask.device
    pos = torch.arange(T, device=device).unsqueeze(0)
    step_region = (pos >= sep_pos.unsqueeze(1)) & (pos < val_len.unsqueeze(1))
    keep = torch.rand(B, T, device=device) >= mask_prob
    masked = attention_mask.clone()
    masked[step_region & ~keep] = 0
    return masked


# ---------------------------------------------------------------------------
# First-token NLL: gradient flows through h_hat_k only
# ---------------------------------------------------------------------------

def per_item_first_token_nll(
    pred_logits: torch.Tensor,   # (B, T, V) — already sliced by sparsity_factor
    pred_ids: torch.Tensor,      # (B, T)
    pred_prefix_lens: list[int], # position of first s_{k+1} token per item
) -> torch.Tensor:
    """Per-item cross-entropy at the first token of s_{k+1}. Returns (B,)."""
    B = pred_logits.shape[0]
    first_logits = torch.stack(
        [pred_logits[i, pred_prefix_lens[i]] for i in range(B)]
    )  # (B, V)
    first_targets = torch.stack(
        [pred_ids[i, pred_prefix_lens[i]] for i in range(B)]
    )  # (B,)
    return F.cross_entropy(first_logits, first_targets, reduction="none")


def pred_first_token_nll_loss(
    pred_logits: torch.Tensor,
    pred_ids: torch.Tensor,
    pred_prefix_lens: list[int],
) -> torch.Tensor:
    """Cross-entropy at the first token of s_{k+1} only (mean over batch).

    At that decoder position the only available context is [recons | pred_prefix],
    which means gradient flows entirely through h_hat_k.  If this NLL decreases
    with the prediction branch enabled, h_hat_k is learning trajectory information.
    """
    return per_item_first_token_nll(pred_logits, pred_ids, pred_prefix_lens).mean()


# ---------------------------------------------------------------------------
# Validation correlation helper
# ---------------------------------------------------------------------------

def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation between two equal-length lists. Returns nan if < 2 points."""
    if len(xs) < 2:
        return float("nan")
    t = torch.tensor([xs, ys], dtype=torch.float32)
    return float(torch.corrcoef(t)[0, 1].item())


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
            "config": {"sparsity_factor": model.sparsity_factor, "phase": "future"},
        },
        path,
    )
    print(f"  [ckpt] {path}  step={step}  val_total={best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_future_ssae(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"Initialising Future-SSAE from {args.model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "<sep>"})
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    model = SSAE(
        tokenizer=tokenizer,
        sparsity_factor=args.sparsity_factor,
        phase=1,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  n_inputs={model.n_inputs}  n_latents={model.n_latents}  "
          f"trainable_params={n_trainable:,}")

    eos_id = tokenizer.eos_token_id
    pad_id = eos_id
    sep_id = tokenizer.sep_token_id
    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    if not space_ids:
        raise ValueError(
            f"Tokenizer {args.model_id} cannot encode a single space; "
            "Future-SSAE collate requires a single-token space separator."
        )
    space_id = space_ids[0]

    print(f"Loading train data from {args.data} …")
    full_ds = GSM8KFutureStepDataset(args.data, tokenizer, max_length=args.max_length)

    if args.val_data:
        train_ds = full_ds
        val_ds = GSM8KFutureStepDataset(args.val_data, tokenizer, max_length=args.max_length)
    else:
        n_val = max(1, int(len(full_ds) * args.val_frac))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )

    collate = GSM8KFutureCollateFn(
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        sep_token_id=sep_id,
        space_token_id=space_id,
        prediction_context_mode=args.pred_context_mode,
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
    print(f"  alpha_pred={args.alpha_pred}  pred_context_mode={args.pred_context_mode}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    total_grad_steps = (len(train_loader) // args.grad_accum) * args.epochs

    dwa = DWAController(
        target=args.l1_target,
        init_weight=args.l1_weight_init,
        update_freq=args.l1_dwa_interval,
    )
    pred_scaler = PredNLLScaler()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "best.pt"

    best_val = float("inf")
    global_step = 0

    print(f"\nFuture-SSAE training: {args.epochs} epochs  "
          f"lr={args.lr:.0e}  l1_target={args.l1_target}  "
          f"alpha_pred={args.alpha_pred}  grad_accum={args.grad_accum}")

    with open(log_path, "w") as log_f:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            accum = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                recon_ids = batch["recon_input_ids"].to(device)
                recon_attn = batch["recon_attention_mask"].to(device)
                recon_loss_mask = batch["recon_loss_mask"].to(device)
                recon_hints_ids = batch["recon_hints_sep_ids"].to(device)
                recon_hints_attn = batch["recon_hints_sep_attn"].to(device)
                sep_pos = torch.as_tensor(batch["recon_sep_pos"], device=device)
                val_len = torch.as_tensor(batch["recon_val_len"], device=device)

                pred_ids = batch["pred_input_ids"].to(device)
                pred_attn = batch["pred_attention_mask"].to(device)
                pred_loss_mask = batch["pred_loss_mask"].to(device)
                pred_prefix_lens = batch["pred_prefix_len"]  # list[int], stays on CPU

                masked_attn = apply_step_attention_mask(
                    recon_attn, sep_pos, val_len, args.mask_prob
                )

                l1_w = dwa.current_weight
                pred_scale = pred_scaler.current_scale

                # Reconstruction branch — encoder + sparse projection + decode
                latents, loss_sparsity, recon_logits = model(
                    recon_ids, masked_attn, recon_hints_ids, recon_hints_attn
                )
                recon_logits = recon_logits[:, args.sparsity_factor - 1:, :]
                nll_recon = masked_nll_loss(recon_logits, recon_ids, recon_loss_mask)

                # Prediction branch — reuse same latents, different decoder context
                pred_logits = model.decode_from_latents(latents, pred_ids, pred_attn)
                pred_logits = pred_logits[:, args.sparsity_factor - 1:, :]
                nll_pred = masked_nll_loss(pred_logits, pred_ids, pred_loss_mask)

                # First-token NLL: no s_{k+1} context at this position, so
                # gradient flows entirely through h_hat_k
                nll_pred_first = pred_first_token_nll_loss(
                    pred_logits, pred_ids, pred_prefix_lens
                )

                loss_spa = loss_sparsity / recon_ids.shape[0]
                # Normalise alpha term so nll_pred / pred_scale ≈ O(1) regardless
                # of how much harder prediction is than reconstruction
                loss = nll_recon + l1_w * loss_spa + args.alpha_pred * (nll_pred / pred_scale)

                (loss / args.grad_accum).backward()
                accum += 1

                dwa.step(loss_spa.item())
                pred_scaler.step(nll_pred.item())

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

                    with torch.no_grad():
                        h = latents.squeeze(1)
                        n_active_frac = (h.abs() > 1e-3).float().mean().item()
                        h_hat_l1_mean = h.abs().mean().item()

                    log_f.write(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": global_step,
                                "nll_recon": nll_recon.item(),
                                "nll_pred": nll_pred.item(),
                                "pred_first_token_nll": nll_pred_first.item(),
                                "sparsity": loss_spa.item(),
                                "loss_total": loss.item(),
                                "l1_weight": l1_w,
                                "alpha_pred": args.alpha_pred,
                                "pred_nll_scale": pred_scale,
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
            val_recon_total = 0.0
            val_pred_total = 0.0
            val_spa_total = 0.0
            val_recon_per_batch: list[float] = []
            val_pred_per_batch: list[float] = []
            # Stratify per-item pred NLL by current_step_idx bucket
            # (0 = first step, 1 = second, 2plus = anything later).
            pred_nll_by_idx: dict[str, list[float]] = {"0": [], "1": [], "2plus": []}
            first_token_nll_by_idx: dict[str, list[float]] = {"0": [], "1": [], "2plus": []}

            with torch.no_grad():
                for batch in val_loader:
                    recon_ids = batch["recon_input_ids"].to(device)
                    recon_attn = batch["recon_attention_mask"].to(device)
                    recon_loss_mask = batch["recon_loss_mask"].to(device)
                    recon_hints_ids = batch["recon_hints_sep_ids"].to(device)
                    recon_hints_attn = batch["recon_hints_sep_attn"].to(device)
                    pred_ids = batch["pred_input_ids"].to(device)
                    pred_attn = batch["pred_attention_mask"].to(device)
                    pred_loss_mask = batch["pred_loss_mask"].to(device)
                    pred_prefix_lens = batch["pred_prefix_len"]
                    step_idxs = batch["current_step_idx"]

                    latents, loss_sparsity, recon_logits = model(
                        recon_ids, recon_attn, recon_hints_ids, recon_hints_attn
                    )
                    recon_logits = recon_logits[:, args.sparsity_factor - 1:, :]
                    batch_recon_nll = masked_nll_loss(recon_logits, recon_ids, recon_loss_mask).item()

                    pred_logits = model.decode_from_latents(latents, pred_ids, pred_attn)
                    pred_logits = pred_logits[:, args.sparsity_factor - 1:, :]
                    batch_pred_nll = masked_nll_loss(pred_logits, pred_ids, pred_loss_mask).item()

                    val_recon_total += batch_recon_nll
                    val_pred_total += batch_pred_nll
                    val_spa_total += (loss_sparsity / recon_ids.shape[0]).item()
                    val_recon_per_batch.append(batch_recon_nll)
                    val_pred_per_batch.append(batch_pred_nll)

                    # Per-item NLLs for stratification
                    item_pred_nll = per_item_masked_nll(
                        pred_logits, pred_ids, pred_loss_mask
                    ).cpu().tolist()
                    item_first_nll = per_item_first_token_nll(
                        pred_logits, pred_ids, pred_prefix_lens
                    ).cpu().tolist()
                    for i, idx in enumerate(step_idxs):
                        bucket = "0" if idx == 0 else "1" if idx == 1 else "2plus"
                        pred_nll_by_idx[bucket].append(item_pred_nll[i])
                        first_token_nll_by_idx[bucket].append(item_first_nll[i])

            n_batches = max(len(val_loader), 1)
            val_recon = val_recon_total / n_batches
            val_pred = val_pred_total / n_batches
            val_spa = val_spa_total / n_batches
            val_total = val_recon + dwa.current_weight * val_spa + args.alpha_pred * val_pred
            corr_recon_pred = pearson_r(val_recon_per_batch, val_pred_per_batch)

            def _safe_mean(xs: list[float]) -> float:
                return sum(xs) / len(xs) if xs else float("nan")

            val_pred_idx0 = _safe_mean(pred_nll_by_idx["0"])
            val_pred_idx1 = _safe_mean(pred_nll_by_idx["1"])
            val_pred_idx2plus = _safe_mean(pred_nll_by_idx["2plus"])
            val_first_idx0 = _safe_mean(first_token_nll_by_idx["0"])
            val_first_idx1 = _safe_mean(first_token_nll_by_idx["1"])
            val_first_idx2plus = _safe_mean(first_token_nll_by_idx["2plus"])

            log_f.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "val_recon_nll": val_recon,
                        "val_pred_nll": val_pred,
                        "val_sparsity": val_spa,
                        "val_total": val_total,
                        "val_corr_recon_pred": corr_recon_pred,
                        "val_pred_nll_idx0": val_pred_idx0,
                        "val_pred_nll_idx1": val_pred_idx1,
                        "val_pred_nll_idx2plus": val_pred_idx2plus,
                        "val_pred_first_token_nll_idx0": val_first_idx0,
                        "val_pred_first_token_nll_idx1": val_first_idx1,
                        "val_pred_first_token_nll_idx2plus": val_first_idx2plus,
                        "val_n_idx0": len(pred_nll_by_idx["0"]),
                        "val_n_idx1": len(pred_nll_by_idx["1"]),
                        "val_n_idx2plus": len(pred_nll_by_idx["2plus"]),
                    }
                )
                + "\n"
            )
            log_f.flush()

            print(
                f"  Epoch {epoch + 1}: val_recon={val_recon:.4f}  val_pred={val_pred:.4f}  "
                f"val_spa={val_spa:.3f}  val_total={val_total:.4f}  "
                f"corr_rp={corr_recon_pred:+.3f}  "
                f"pred_scale={pred_scaler.current_scale:.3f}  "
                f"l1w={dwa.current_weight:.2e}  best={best_val:.4f}\n"
                f"           pred_nll[idx]:  "
                f"0={val_pred_idx0:.3f}  1={val_pred_idx1:.3f}  2+={val_pred_idx2plus:.3f}  "
                f"|  first_tok[idx]:  "
                f"0={val_first_idx0:.3f}  1={val_first_idx1:.3f}  2+={val_first_idx2plus:.3f}"
            )

            if val_total < best_val:
                best_val = val_total
                save_checkpoint(ckpt_path, model, optimizer, global_step, best_val)

    print(f"\nDone. Best val_total={best_val:.4f}  →  {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Future-SSAE training on GSM8K-Aug")
    # Data
    p.add_argument("--data", required=True, help="Path to train JSONL")
    p.add_argument("--val-data", default=None, help="Separate val JSONL (else auto-split)")
    p.add_argument("--output-dir", required=True, help="Dir for checkpoints + train_log.jsonl")
    # Model
    p.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--sparsity-factor", type=int, default=1)
    # Sparsity
    p.add_argument("--l1-target", type=float, default=3.0)
    p.add_argument("--l1-weight-init", type=float, default=1e-4)
    p.add_argument("--l1-dwa-interval", type=int, default=100)
    # Future-SSAE specific
    p.add_argument("--alpha-pred", type=float, default=0.1,
                   help="Weight for the next-step prediction loss (applied after EMA normalisation). "
                        "Recommended sweep: 0.03, 0.1, 0.3, 1.0. "
                        "Monitor val_corr_recon_pred: if it goes below -0.5, alpha is too large.")
    p.add_argument("--pred-context-mode", default="q_current",
                   choices=list(GSM8KFutureCollateFn.MODES),
                   help="Context given to the prediction decoder. "
                        "Recommended: start with 'q_current', then 'q_prev1_current'. "
                        "'q_prev2_current' and 'full_context_current' are available but "
                        "not recommended until q_current/q_prev1 baselines are established.")
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
    train_future_ssae(parse_args())
