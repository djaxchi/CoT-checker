#!/usr/bin/env python3
"""Train the SSAE on symbolic logic reasoning traces.

Implements the three-phase training recipe from arXiv:2603.03031 (Miaow-Lab):

  Phase 1 — Reconstruction
    Trains encoder + autoencoder + projection_mlp + decoder to reconstruct
    step tokens from a sparse latent h_c.
    Loss: NLL_reconstruction + L1_weight * sparsity
    L1_weight is adapted by DWA to keep sparsity around L1_TARGET.

  Phase 2 — Distribution (last-token)
    Freezes phase-1 modules; trains hints_encoder + mean_mlp + var_mlp to
    predict the phase-1 latent from context alone (last-token pooling).
    Loss: Gaussian NLL between predicted and encoder-derived latents.

  Phase 3 — Distribution (avg-token)
    Same as phase 2 but hints_encoder uses average-pooling.
    Only var_mlp is trained.

Input: a JSONL traces file produced by generate_symbolic_data.py.
Each line: {"question": ..., "query": ..., "steps": [...]}

Usage:
    python scripts/train_ssae_symbolic.py \\
        --traces data/prontoqa_traces.jsonl \\
        --output results/checkpoints/ssae_symbolic.pt \\
        --phase 1 \\
        --device mps

    # Resume from checkpoint and run phase 2
    python scripts/train_ssae_symbolic.py \\
        --traces data/prontoqa_traces.jsonl \\
        --output results/checkpoints/ssae_symbolic_p2.pt \\
        --resume results/checkpoints/ssae_symbolic.pt \\
        --phase 2 --device mps
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.saes.ssae import SSAE

# ---------------------------------------------------------------------------
# Hyper-parameters (matching paper defaults)
# ---------------------------------------------------------------------------

PHASE_CFG = {
    # decay_iters is overridden at runtime to match actual gradient steps
    # (n_batches // grad_accum * epochs) — the values here are fallback defaults
    1: dict(lr=1e-6,  min_lr=1e-7,  warmup=100, decay_iters=None, epochs=1),
    2: dict(lr=1e-4,  min_lr=1e-5,  warmup=20,  decay_iters=None, epochs=3),
    3: dict(lr=1e-4,  min_lr=1e-5,  warmup=20,  decay_iters=None, epochs=1),
}

L1_WEIGHT_INIT  = 1e-4   # initial L1 penalty on sparse latents
L1_TARGET       = 3.0    # DWA target: keep sparsity loss ≈ this value
L1_DWA_INTERVAL = 100    # update L1_weight every N optimizer steps
GRAD_CLIP       = 1.0
MASK_PROB       = 0.10   # fraction of step tokens randomly masked in Phase 1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SymbolicTracesDataset(Dataset):
    """One sample = one (context, step) pair from a traces JSONL.

    Expands each problem trace into one record per step:
        context = question + query + all prior steps
        step    = current step text
    """

    def __init__(self, path: str | Path, tokenizer, max_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: list[dict] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                question = item["question"]
                query    = item.get("query", "")
                steps    = item["steps"]
                for i, step in enumerate(steps):
                    prior   = " ".join(steps[:i])
                    context = f"{question} {query} {prior}".strip()
                    self.records.append({"context": context, "step": step})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        ctx_ids  = self.tokenizer.encode(rec["context"], max_length=self.max_length, truncation=True)
        step_ids = self.tokenizer.encode(rec["step"],    max_length=self.max_length, truncation=True)
        return {
            "context_ids": torch.tensor(ctx_ids,  dtype=torch.long),
            "step_ids":    torch.tensor(step_ids, dtype=torch.long),
        }


class SSAECollateFn:
    """Build SSAE training batches from (context_ids, step_ids) pairs.

    Constructs:
        input_ids               (B, max_seq)  — [context | sep | step | eos]
        attention_mask          (B, max_seq)  — 1 for real tokens
        loss_mask               (B, max_seq)  — 1 for step tokens only (post-sep)
        hints_sep_ids           (B, max_hint) — [context | sep]
        hints_sep_attention_masks (B, max_hint)
        sep_pos                 list[int]     — sep position per sample
    """

    def __init__(self, eos_id: int, pad_id: int, sep_id: int) -> None:
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.sep_id = sep_id

    def __call__(self, batch: list[dict]) -> dict:
        full_seqs, sep_positions, hint_seqs = [], [], []

        for rec in batch:
            ctx  = rec["context_ids"]
            step = rec["step_ids"]
            sep_pos = len(ctx)
            seq = torch.cat([ctx,
                             torch.tensor([self.sep_id]),
                             step,
                             torch.tensor([self.eos_id])])
            full_seqs.append(seq)
            sep_positions.append(sep_pos)
            hint_seqs.append(seq[:sep_pos + 1])   # context + sep

        max_seq  = max(len(s) for s in full_seqs)
        max_hint = max(len(h) for h in hint_seqs)

        input_ids_list, attn_list, loss_list  = [], [], []
        hint_ids_list,  hint_attn_list        = [], []

        for i, seq in enumerate(full_seqs):
            pad = max_seq - len(seq)
            ids   = torch.cat([seq, torch.full((pad,), self.pad_id)])
            attn  = torch.cat([torch.ones(len(seq)), torch.zeros(pad)]).long()
            # loss mask: 1 from (sep_pos+1) to end of real tokens
            lm    = torch.zeros(max_seq, dtype=torch.long)
            lm[sep_positions[i] + 1 : len(seq)] = 1

            input_ids_list.append(ids)
            attn_list.append(attn)
            loss_list.append(lm)

            h    = hint_seqs[i]
            hpad = max_hint - len(h)
            hint_ids_list.append(
                torch.cat([h, torch.full((hpad,), self.pad_id)]))
            hint_attn_list.append(
                torch.cat([torch.ones(len(h)), torch.zeros(hpad)]).long())

        return {
            "input_ids":                  torch.stack(input_ids_list),
            "attention_mask":             torch.stack(attn_list),
            "loss_mask":                  torch.stack(loss_list),
            "hints_sep_ids":              torch.stack(hint_ids_list),
            "hints_sep_attention_masks":  torch.stack(hint_attn_list),
            "sep_pos":                    sep_positions,
        }


# ---------------------------------------------------------------------------
# Learning-rate schedule (linear warmup + cosine annealing)
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
# NLL loss (Phase 1)
# ---------------------------------------------------------------------------

def phase1_nll_loss(
    logits:    torch.Tensor,   # (B, seq_len, V)
    input_ids: torch.Tensor,   # (B, seq_len)
    loss_mask: torch.Tensor,   # (B, seq_len)  1 = step tokens
) -> torch.Tensor:
    """Cross-entropy on step tokens only.

    With sparsity_factor=1, the decoder produces logits of the same length as
    input_ids because it sees [recon_0 | x_0 .. x_{N-2}] and logits[i] predicts
    the token at position i in input_ids.
    """
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.view(-1, V), input_ids.view(-1), reduction="none")
    ce = ce.view(B, T)
    denom = loss_mask.float().sum().clamp(min=1)
    return (ce * loss_mask.float()).sum() / denom


# ---------------------------------------------------------------------------
# Input masking (Phase 1) — randomly zero out 10% of step token ids
# ---------------------------------------------------------------------------

def apply_step_mask(
    input_ids:  torch.Tensor,   # (B, seq_len)
    loss_mask:  torch.Tensor,   # (B, seq_len)
    mask_prob:  float,
    mask_token: int,
) -> torch.Tensor:
    """Replace mask_prob fraction of step tokens with mask_token."""
    if mask_prob <= 0:
        return input_ids
    masked = input_ids.clone()
    rand  = torch.rand_like(input_ids, dtype=torch.float)
    where = (rand < mask_prob) & (loss_mask == 1)
    masked[where] = mask_token
    return masked


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model: SSAE, optimizer, step: int,
                    best_val_loss: float, cfg: dict,
                    frozen_backbones: bool = False) -> None:
    """Save two files:

    <path>          — training checkpoint (resume-able): all trainable weights
                      + optimizer state. Backbone excluded when frozen (~30 MB).

    <path>.enc      — inference-only checkpoint: just the autoencoder weights
                      (~6 MB). This is all that's needed to call model.encode().
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    _BACKBONE_PREFIXES = ("encoder.", "decoder.", "hints_encoder.")

    if frozen_backbones:
        train_state = {k: v for k, v in model.state_dict().items()
                       if not any(k.startswith(p) for p in _BACKBONE_PREFIXES)}
    else:
        train_state = model.state_dict()

    # Training checkpoint (resume)
    torch.save({
        "model":            train_state,
        "frozen_backbones": frozen_backbones,
        "optimizer":        optimizer.state_dict(),
        "step":             step,
        "best_val_loss":    best_val_loss,
        "config":           cfg,
        "encoder_name":     "Qwen/Qwen2.5-0.5B",
        "decoder_name":     "Qwen/Qwen2.5-0.5B",
        "sparsity_factor":  model.sparsity_factor,
    }, path)
    print(f"  [ckpt] saved → {path}  (step={step}, val_loss={best_val_loss:.4f})")

    # Inference-only checkpoint: just the autoencoder (encoder + decoder halves)
    enc_path = path.with_suffix(".enc.pt")
    torch.save({
        "autoencoder":     {k.removeprefix("autoencoder."): v
                            for k, v in model.state_dict().items()
                            if k.startswith("autoencoder.")},
        "encoder_name":    "Qwen/Qwen2.5-0.5B",
        "sparsity_factor": model.sparsity_factor,
        "best_val_loss":   best_val_loss,
        "step":            step,
    }, enc_path)
    size_mb = enc_path.stat().st_size / 1e6
    print(f"  [enc]  saved → {enc_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Tokenizer + model ---
    if args.resume:
        print(f"Resuming from {args.resume} (phase {args.phase}) …")
        model = SSAE.from_checkpoint(args.resume, device=device, phase=args.phase)
    else:
        print("Initialising fresh SSAE (Qwen2.5-0.5B, float32) …")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        if tokenizer.sep_token is None:
            tokenizer.add_special_tokens({"sep_token": "<sep>"})
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        # Force float32 throughout — MPS rejects mixed bfloat16/float32 matmuls
        model = SSAE(
            tokenizer=tokenizer, sparsity_factor=1, phase=args.phase,
            dtype=torch.float32,
        ).to(device)

    # ------------------------------------------------------------------
    # Backbone freezing (memory optimisation for MPS / single-GPU)
    #
    # The full paper trains encoder + decoder jointly with the autoencoder.
    # On MPS (≤20 GB), 3 × Qwen2.5-0.5B + AdamW moments ≈ 20 GB — OOM.
    # Freezing the LLM backbones and only training the sparse projection
    # (autoencoder + projection_mlp) reduces trainable params from ~1B to
    # ~1.6 M and cuts optimizer memory by ~16 GB.  The backbone already
    # encodes symbolic reasoning; the SAE learns a sparse basis on top.
    # ------------------------------------------------------------------
    freeze_backbones = args.freeze_backbones
    if freeze_backbones:
        for name, param in model.named_parameters():
            if any(bb in name for bb in ("encoder.", "decoder.", "hints_encoder.")):
                param.requires_grad_(False)
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Backbones frozen. Trainable modules: {set(n.split('.')[0] for n in trainable_names)}")
        print(f"  Trainable params: {n_trainable:,}")

    tokenizer = model.tokenizer
    vocab_size = len(tokenizer)
    sep_id = tokenizer.sep_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = eos_id  # Qwen uses eos as pad

    # --- Dataset ---
    print(f"Loading traces from {args.traces} …")
    full_ds = SymbolicTracesDataset(args.traces, tokenizer, max_length=args.max_length)
    n_val   = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    collate = SSAECollateFn(eos_id=eos_id, pad_id=pad_id, sep_id=sep_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate, drop_last=False,
    )

    print(f"  Train: {n_train} steps  |  Val: {n_val} steps")
    print(f"  Batches/epoch: {len(train_loader)}  |  Phase: {args.phase}")

    # --- Optimizer ---
    cfg = dict(PHASE_CFG[args.phase])  # copy so we can mutate

    # Set decay_iters to span the full training run so cosine annealing
    # reaches min_lr only at the very last gradient step, not after 30 steps.
    total_epochs = args.epochs if args.epochs else cfg["epochs"]
    total_grad_steps = (len(train_loader) // args.grad_accum) * total_epochs
    cfg["decay_iters"] = total_grad_steps
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found. Check phase + freeze settings.")
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg["lr"],
        betas=(0.9, 0.95), weight_decay=0.01,
    )

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass  # shape mismatch on phase switch — start fresh optimizer

    # --- Training state ---
    l1_weight    = L1_WEIGHT_INIT
    best_val     = float("inf")
    global_step  = 0
    accum_steps  = args.grad_accum
    total_epochs = args.epochs if args.epochs else cfg["epochs"]

    print(f"\nStarting Phase {args.phase} training for {total_epochs} epoch(s)")
    print(f"  LR={cfg['lr']:.0e}  L1_weight={l1_weight:.0e}  grad_accum={accum_steps}")

    for epoch in range(total_epochs):
        model.train()
        optimizer.zero_grad()
        running_nll = running_spa = 0.0
        n_accum = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids     = batch["input_ids"].to(device)
            attention_mask= batch["attention_mask"].to(device)
            loss_mask     = batch["loss_mask"].to(device)
            hints_sep_ids = batch["hints_sep_ids"].to(device)
            hints_sep_attn= batch["hints_sep_attention_masks"].to(device)

            # --- Forward ---
            if args.phase == 1:
                masked_ids = apply_step_mask(
                    input_ids, loss_mask, MASK_PROB, mask_token=pad_id
                )
                latents, loss_sparsity, logits = model(
                    masked_ids, attention_mask, hints_sep_ids, hints_sep_attn
                )
                loss_nll = phase1_nll_loss(logits, input_ids, loss_mask)
                loss_spa = loss_sparsity / input_ids.shape[0]
                loss = loss_nll + l1_weight * loss_spa

                running_nll += loss_nll.item()
                running_spa += loss_spa.item()

            else:  # phase 2 or 3
                nll_loss, mean_error = model(
                    input_ids, attention_mask, hints_sep_ids, hints_sep_attn
                )
                loss = nll_loss
                running_nll += nll_loss.item()

            # --- Backward + gradient accumulation ---
            (loss / accum_steps).backward()
            n_accum += 1

            if n_accum == accum_steps:
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)

                # LR schedule
                new_lr = get_lr(global_step, cfg["lr"], cfg["min_lr"],
                                 cfg["warmup"], cfg["decay_iters"])
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                n_accum = 0

                # DWA: adjust L1_weight every L1_DWA_INTERVAL steps (Phase 1 only)
                if args.phase == 1 and global_step % L1_DWA_INTERVAL == 0:
                    if running_spa / max(L1_DWA_INTERVAL, 1) > L1_TARGET:
                        l1_weight = min(l1_weight * 1.01, 0.1)
                    else:
                        l1_weight = max(l1_weight * 0.99, 1e-6)
                    running_spa = 0.0
                    running_nll = 0.0

            if batch_idx % 50 == 0:
                avg_nll = running_nll / max(batch_idx + 1, 1)
                if args.phase == 1:
                    avg_spa = running_spa / max(batch_idx + 1, 1)
                    pbar.set_postfix(nll=f"{avg_nll:.4f}", spa=f"{avg_spa:.3f}",
                                     l1w=f"{l1_weight:.1e}", lr=f"{new_lr:.1e}" if global_step > 0 else "warmup")
                else:
                    pbar.set_postfix(nll=f"{avg_nll:.4f}")

        # --- Validation ---
        model.eval()
        val_nll = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                loss_mask      = batch["loss_mask"].to(device)
                hints_sep_ids  = batch["hints_sep_ids"].to(device)
                hints_sep_attn = batch["hints_sep_attention_masks"].to(device)

                if args.phase == 1:
                    latents, loss_sparsity, logits = model(
                        input_ids, attention_mask, hints_sep_ids, hints_sep_attn
                    )
                    val_nll += phase1_nll_loss(logits, input_ids, loss_mask).item()
                else:
                    nll_loss, _ = model(
                        input_ids, attention_mask, hints_sep_ids, hints_sep_attn
                    )
                    val_nll += nll_loss.item()

        val_nll /= max(len(val_loader), 1)
        print(f"\nEpoch {epoch+1} — val_loss_nll: {val_nll:.4f}  (best: {best_val:.4f})")

        if val_nll < best_val:
            best_val = val_nll
            save_checkpoint(
                Path(args.output), model, optimizer, global_step, best_val,
                cfg={"sparsity_factor": 1, "phase": args.phase},
                frozen_backbones=freeze_backbones,
            )

    # Save final regardless
    final_path = Path(args.output).with_stem(Path(args.output).stem + "_final")
    save_checkpoint(final_path, model, optimizer, global_step, best_val,
                    cfg={"sparsity_factor": 1, "phase": args.phase},
                    frozen_backbones=freeze_backbones)
    print(f"\nTraining complete. Best val_loss: {best_val:.4f}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces",     required=True, help="JSONL traces file")
    p.add_argument("--output",     required=True, help="Path to save best checkpoint (.pt)")
    p.add_argument("--resume",     default=None,  help="Checkpoint to resume from")
    p.add_argument("--phase",      type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--epochs",     type=int, default=None,
                   help="Override default epochs for the phase")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-device batch size (MPS: 4 recommended)")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--val-frac",   type=float, default=0.05)
    p.add_argument("--device",     default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument(
        "--freeze-backbones", action=argparse.BooleanOptionalAction, default=True,
        help="Freeze LLM backbones; only train autoencoder + projection_mlp. "
             "Required on MPS (≤20 GB). Disable on A100/H100 to match paper exactly.",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
