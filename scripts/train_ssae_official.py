"""Phase-1 SSAE training, DDP-ready (torchrun --nproc_per_node=4 ...).

Mirrors papers/SSAE/train.py phase-1 loss alignment:

    logits = model(input_ids, attention_mask)[3]                # (B, S+sf-1, V)
    logits_for_labels = logits[:, sparsity_factor - 1:, :]       # (B, S, V)
    ce = CrossEntropyLoss(reduction='none')(...) * loss_mask     # CE on step tokens
    loss_nll = ce.sum() / loss_mask.sum()
    loss_sparsity = sparsity_loss / B
    loss = loss_nll + l1_weight * loss_sparsity
    # ssae_contrastive only:
    loss += bce_weight * BCEWithLogits(aux_logit, label)

Single-GPU is supported (RANK unset) for smoke/debug. Production must use
torchrun + 4 H100 GPUs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ssae.dataset import (  # noqa: E402
    STEP_SEP_TOKEN,
    SSAECollator,
    SSAEJsonlDataset,
    add_step_sep_token,
)
from src.ssae.model_qwen_ssae import QwenSSAE  # noqa: E402


METHODS = ("ssae_positive", "ssae_mixed", "ssae_contrastive")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--train_jsonl", required=True, type=Path)
    p.add_argument("--val_jsonl", required=True, type=Path,
                   help="Held-out JSONL used for reconstruction val loss "
                        "(this is the SAME prm800k_val_1k.jsonl used later "
                        "for probe threshold selection).")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--phase", type=int, default=1)
    p.add_argument("--sparsity_factor", type=int, default=1)
    p.add_argument("--l1_weight", type=float, default=1e-4)
    p.add_argument("--bce_weight", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Per-GPU micro-batch size.")
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--warmup_iters", type=int, default=2)
    p.add_argument("--max_iters", type=int, default=30)
    p.add_argument("--lr_decay_iters", type=int, default=-1,
                   help="Defaults to max_iters when <= 0.")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ddp_backend", default="nccl")
    p.add_argument("--torch_compile", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--eval_interval_iters", type=int, default=1)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: limit train to --smoke_train_n and "
                        "val to --smoke_val_n.")
    p.add_argument("--smoke_train_n", type=int, default=128)
    p.add_argument("--smoke_val_n", type=int, default=32)
    p.add_argument("--length_audit_only", action="store_true",
                   help="Tokenize all rows, report length stats, and exit. "
                        "Must be run single-process (NOT under torchrun/DDP).")
    p.add_argument("--train_attn_mask_ratio", type=float, default=0.1,
                   help="Random masking ratio applied to attention_mask on step "
                        "tokens during training only (mirror of official phase-1 "
                        "papers/SSAE/train.py mask=True path). Default 0.1 for "
                        "faithful SSAE reproduction. Set to 0 to disable.")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable HF gradient checkpointing on encoder and "
                        "decoder. Recommended for Qwen2.5-1.5B SSAE training.")
    p.add_argument("--ce_chunk_size", type=int, default=2048,
                   help="Max number of active tokens per CE chunk. Larger "
                        "values reduce loop overhead; smaller values reduce "
                        "peak intermediate memory inside cross_entropy.")
    return p.parse_args()


def git_commit_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_lr(it: int, warmup_iters: int, lr_decay_iters: int,
           learning_rate: float, min_lr: float) -> float:
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(lr_decay_iters - warmup_iters, 1)
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def setup_ddp(backend: str) -> tuple[bool, int, int, int, str]:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        return True, rank, local_rank, world_size, device
    rank, local_rank, world_size = 0, 0, 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return False, rank, local_rank, world_size, device


def maybe_log(master: bool, logger, msg: str) -> None:
    if master:
        logger.info(msg)


def _chunked_ce_sum(active_logits: torch.Tensor, active_labels: torch.Tensor,
                    chunk_size: int) -> torch.Tensor:
    """Sum of per-token CE over active positions, computed in chunks.

    Mathematically equivalent to F.cross_entropy(active_logits, active_labels,
    reduction='sum'): summation just splits into contiguous chunks. Reduces
    peak memory inside log_softmax/softmax intermediates.
    """
    n = active_logits.size(0)
    if n == 0:
        return torch.zeros((), device=active_logits.device, dtype=torch.float32)
    if chunk_size <= 0 or n <= chunk_size:
        return F.cross_entropy(active_logits, active_labels, reduction="sum")
    total = torch.zeros((), device=active_logits.device, dtype=torch.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        total = total + F.cross_entropy(
            active_logits[start:end], active_labels[start:end], reduction="sum"
        )
    return total


def compute_loss(model, batch, device, sparsity_factor: int,
                 l1_weight: float, bce_weight: float, contrastive: bool,
                 train_attn_mask_ratio: float = 0.0,
                 ce_chunk_size: int = 2048):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    loss_mask = batch["loss_mask"].to(device, non_blocking=True)

    # Mirror of papers/SSAE/train.py phase-1 mask=True path: during training
    # only, randomly drop attention to step tokens (between sep_pos and
    # val_len) with probability train_attn_mask_ratio. Disabled at eval time
    # (caller passes train_attn_mask_ratio=0.0).
    if train_attn_mask_ratio > 0:
        sep_pos = batch["sep_pos"].to(device, non_blocking=True)
        val_len = batch["val_len"].to(device, non_blocking=True)
        bsz, seq_len = attention_mask.shape
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        interval = (pos >= sep_pos.unsqueeze(1)) & (pos < val_len.unsqueeze(1))
        keep = torch.rand(bsz, seq_len, device=device) >= train_attn_mask_ratio
        attention_mask = attention_mask & (~interval | keep).to(attention_mask.dtype)

    _latents, _latents_clean, sparsity_loss_sum, logits, aux_logit = model(
        input_ids, attention_mask
    )

    # Align logits with input_ids per spec section 9.
    logits_for_labels = logits[:, sparsity_factor - 1:, :]
    if logits_for_labels.shape[1] != input_ids.shape[1]:
        raise RuntimeError(
            "Logit/label alignment failed: "
            f"logits_for_labels {tuple(logits_for_labels.shape)} vs "
            f"input_ids {tuple(input_ids.shape)}"
        )

    # Memory-efficient CE: gather active (step + eos, non-pad) positions before
    # log_softmax so the V-wide softmax is only computed over loss-relevant
    # tokens. Equivalent to (per-token CE * loss_mask).sum() / loss_mask.sum().
    vocab_size = logits_for_labels.size(-1)
    flat_logits = logits_for_labels.reshape(-1, vocab_size)
    flat_labels = input_ids.reshape(-1)
    active_mask = loss_mask.reshape(-1).bool()
    active_logits = flat_logits[active_mask]
    active_labels = flat_labels[active_mask]
    n_active = active_logits.size(0)
    n_loss_tokens = torch.tensor(max(n_active, 1), device=device, dtype=torch.float32)
    if n_active == 0:
        loss_nll = torch.zeros((), device=device, dtype=torch.float32)
    else:
        ce_sum = _chunked_ce_sum(active_logits, active_labels, ce_chunk_size)
        loss_nll = ce_sum / n_loss_tokens

    batch_size = input_ids.size(0)
    loss_spa = sparsity_loss_sum / batch_size  # mean L1 per example

    loss = loss_nll + l1_weight * loss_spa

    aux_bce = torch.tensor(0.0, device=device)
    if contrastive:
        labels = batch["labels"].to(device, non_blocking=True)
        if (labels < 0).any():
            raise RuntimeError(
                "ssae_contrastive requires every training row to have a "
                "valid 0/1 label; found -1."
            )
        if aux_logit is None:
            raise RuntimeError("Aux head missing although method=ssae_contrastive.")
        aux_bce = F.binary_cross_entropy_with_logits(
            aux_logit.squeeze(-1), labels.float()
        )
        loss = loss + bce_weight * aux_bce

    return {
        "loss": loss,
        "loss_nll": loss_nll.detach(),
        "loss_spa": loss_spa.detach(),
        "aux_bce": aux_bce.detach(),
        "sparsity_sum": sparsity_loss_sum.detach(),
        "n_loss_tokens": n_loss_tokens.detach(),
        "batch_size": batch_size,
    }


def main() -> None:
    args = parse_args()

    contrastive = (args.method == "ssae_contrastive")
    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters > 0 else args.max_iters

    # --length_audit_only must not run under torchrun/DDP: a rank-0 sys.exit()
    # would strand the other ranks. Refuse early with a clear message.
    if args.length_audit_only and int(os.environ.get("RANK", -1)) != -1:
        sys.stderr.write(
            "[train_ssae_official] --length_audit_only must be run "
            "single-process, not under torchrun/DDP.\n"
            "  fix: drop the torchrun launcher, e.g.\n"
            "    python scripts/train_ssae_official.py --length_audit_only ...\n"
        )
        sys.exit(3)

    ddp, rank, local_rank, world_size, device = setup_ddp(args.ddp_backend)
    master = (rank == 0)
    seed_offset = rank
    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)

    log_path = args.out_dir / "logs" / f"train_rank{rank}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("ssae-train")

    commit_hash = git_commit_hash()
    if master:
        cfg_path = args.out_dir / "config.yaml"
        cfg = vars(args).copy()
        cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
        cfg["ddp_world_size"] = world_size
        cfg["effective_global_batch"] = args.batch_size * args.grad_accum_steps * world_size
        cfg["git_commit_hash"] = commit_hash
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True))
        logger.info(f"Wrote config to {cfg_path} (git={commit_hash[:12]})")

    # ---- Tokenizer & special token ----------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_step_sep_token(tokenizer)
    pad_token_id = int(tokenizer.pad_token_id)

    # ---- Datasets ---------------------------------------------------------
    train_ds = SSAEJsonlDataset(args.train_jsonl, tokenizer, max_seq_len=args.max_seq_len)
    val_ds = SSAEJsonlDataset(args.val_jsonl, tokenizer, max_seq_len=args.max_seq_len)

    if master:
        logger.info(f"train rows: {len(train_ds)}; val rows: {len(val_ds)}")

    if args.length_audit_only:
        if master:
            tr_report = train_ds.length_audit(raise_on_violation=False)
            va_report = val_ds.length_audit(raise_on_violation=False)
            audit_path = args.out_dir / "length_audit.json"
            audit_path.write_text(json.dumps(
                {"train": tr_report, "val": va_report}, indent=2
            ))
            logger.info(f"Wrote length audit to {audit_path}")
            if tr_report["n_violations"] or va_report["n_violations"]:
                logger.error("Length violations present; refusing to train.")
                sys.exit(2)
        if ddp:
            destroy_process_group()
        return

    # Strict pre-flight: raise on any overflow before allocating models.
    if master:
        train_ds.length_audit(raise_on_violation=True)
        val_ds.length_audit(raise_on_violation=True)

    if args.smoke:
        train_ds = Subset(train_ds, list(range(min(args.smoke_train_n, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(args.smoke_val_n, len(val_ds)))))
        if master:
            logger.info(f"SMOKE mode: train={len(train_ds)} val={len(val_ds)}")

    collator = SSAECollator(pad_token_id=pad_token_id)

    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ---- Model ------------------------------------------------------------
    model = QwenSSAE(
        tokenizer=tokenizer,
        model_name_or_path=args.model_name_or_path,
        sparsity_factor=args.sparsity_factor,
        phase=1,
        local_files_only=args.local_files_only,
        contrastive=contrastive,
    ).to(device)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if master:
            logger.info("Enabled gradient checkpointing on encoder + decoder; "
                        "decoder use_cache=False.")

    if torch.cuda.is_available():
        mem_after_load = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"[mem] rank {rank} after model load: max_allocated={mem_after_load:.2f} GB")

    if args.torch_compile:
        model = torch.compile(model)

    if ddp:
        # find_unused_parameters=True because hints_encoder is loaded but
        # has no gradient path in phase-1 forward (matches official spec).
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        model_module = model.module
    else:
        model_module = model

    # ---- Optimizer / AMP --------------------------------------------------
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if device != "cpu" and torch.cuda.is_available()
        else nullcontext()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    if master:
        n_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"trainable params: {n_trainable / 1e6:.2f}M")

    # ---- Training loop ----------------------------------------------------
    train_metrics_history: list[dict] = []
    val_metrics_history: list[dict] = []
    best_val_recon_ce = float("inf")
    t_start = time.time()
    examples_seen = 0
    tokens_seen = 0
    global_step = 0

    for it in range(args.max_iters):
        if ddp:
            train_sampler.set_epoch(it)
        lr = get_lr(it, args.warmup_iters, lr_decay_iters,
                    args.learning_rate, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_nll, epoch_spa, epoch_aux, epoch_tokens, n_micro = 0.0, 0.0, 0.0, 0, 0
        micro_step = 0
        for batch in train_loader:
            sync_grads = (micro_step == args.grad_accum_steps - 1) or (not ddp)
            ctx = model.no_sync() if (ddp and not sync_grads) else nullcontext()
            with ctx:
                with autocast_ctx:
                    out = compute_loss(
                        model, batch, device, args.sparsity_factor,
                        args.l1_weight, args.bce_weight, contrastive,
                        train_attn_mask_ratio=args.train_attn_mask_ratio,
                        ce_chunk_size=args.ce_chunk_size,
                    )
                # Scale loss for accumulation
                loss_scaled = out["loss"] / args.grad_accum_steps
                scaler.scale(loss_scaled).backward()

            # Log peak memory once after the very first backward on every rank.
            if global_step == 0 and micro_step == 0 and torch.cuda.is_available():
                mem_after_step1 = torch.cuda.max_memory_allocated() / 1e9
                logger.info(
                    f"[mem] rank {rank} after first fwd+bwd: "
                    f"max_allocated={mem_after_step1:.2f} GB"
                )

            epoch_nll += out["loss_nll"].item()
            epoch_spa += out["loss_spa"].item()
            epoch_aux += out["aux_bce"].item()
            epoch_tokens += int(out["n_loss_tokens"].item())
            examples_seen += out["batch_size"] * world_size
            tokens_seen += int(out["n_loss_tokens"].item()) * world_size
            n_micro += 1
            micro_step += 1

            if micro_step == args.grad_accum_steps:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0
                global_step += 1

        # Flush any trailing micro-steps so no batch is wasted.
        if micro_step > 0:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            global_step += 1

        mean_nll = epoch_nll / max(n_micro, 1)
        mean_spa = epoch_spa / max(n_micro, 1)
        mean_aux = epoch_aux / max(n_micro, 1) if contrastive else None
        if master:
            logger.info(
                f"iter {it} lr {lr:.2e} train_nll {mean_nll:.4f} "
                f"train_l1_mean {mean_spa:.4f}"
                + (f" train_aux_bce {mean_aux:.4f}" if mean_aux is not None else "")
            )
        train_metrics_history.append({
            "iter": it, "lr": lr, "loss_nll": mean_nll, "loss_spa": mean_spa,
            "aux_bce": mean_aux, "n_loss_tokens_in_epoch": epoch_tokens,
        })

        # ---- Validation ---------------------------------------------------
        if (it % args.eval_interval_iters == 0) and master:
            model.eval()
            v_nll, v_spa, v_aux, v_count = 0.0, 0.0, 0.0, 0
            with torch.no_grad(), autocast_ctx:
                for vbatch in val_loader:
                    vout = compute_loss(
                        model, vbatch, device, args.sparsity_factor,
                        args.l1_weight, args.bce_weight,
                        contrastive=False,  # do not require labels on val
                        train_attn_mask_ratio=0.0,  # mask disabled for eval
                        ce_chunk_size=args.ce_chunk_size,
                    )
                    v_nll += vout["loss_nll"].item()
                    v_spa += vout["loss_spa"].item()
                    v_aux += vout["aux_bce"].item()
                    v_count += 1
            v_nll /= max(v_count, 1)
            v_spa /= max(v_count, 1)
            v_aux /= max(v_count, 1)
            logger.info(f"iter {it} val_recon_ce {v_nll:.4f} val_l1_mean {v_spa:.4f}")
            val_metrics_history.append({
                "iter": it, "val_recon_ce": v_nll, "val_l1_mean": v_spa,
            })
            if v_nll < best_val_recon_ce:
                best_val_recon_ce = v_nll
                ckpt_path = args.out_dir / "checkpoints" / "best_val_reconstruction.pt"
                torch.save({
                    "iter": it,
                    "model_state": model_module.state_dict(),
                    "config": vars(args),
                    "val_recon_ce": v_nll,
                }, ckpt_path)
                logger.info(f"Saved best checkpoint to {ckpt_path}")

    train_time = time.time() - t_start

    # ---- Save final artifacts --------------------------------------------
    if master:
        last_ckpt = args.out_dir / "checkpoints" / "last.pt"
        torch.save({
            "iter": args.max_iters - 1,
            "model_state": model_module.state_dict(),
            "config": vars(args),
        }, last_ckpt)
        # Also save a copy named ssae_model.pt (the canonical name per spec).
        torch.save(model_module.state_dict(), args.out_dir / "ssae_model.pt")

        # Aggregate final-iteration metrics.
        last_train = train_metrics_history[-1] if train_metrics_history else {}
        last_val = val_metrics_history[-1] if val_metrics_history else {}
        max_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""

        train_metrics = {
            "method": args.method,
            "reconstruction_ce": last_train.get("loss_nll"),
            "l1_sum": (last_train.get("loss_spa") or 0.0) * args.batch_size,
            "l1_mean": last_train.get("loss_spa"),
            "aux_bce": last_train.get("aux_bce"),
            "total_loss": (
                (last_train.get("loss_nll") or 0.0)
                + args.l1_weight * (last_train.get("loss_spa") or 0.0)
                + (
                    args.bce_weight * (last_train.get("aux_bce") or 0.0)
                    if contrastive else 0.0
                )
            ),
            "learning_rate": last_train.get("lr"),
            "examples_seen": examples_seen,
            "tokens_seen": tokens_seen,
            "train_time_sec": train_time,
            "avg_train_latency_ms_per_example": (
                train_time * 1000.0 / max(examples_seen, 1)
            ),
            "max_memory_allocated_gb": max_mem,
            "gpu_name": gpu_name,
            "n_inputs": model_module.n_inputs,
            "n_latents": model_module.n_latents,
            "sparsity_factor": args.sparsity_factor,
            "model_name_or_path": args.model_name_or_path,
            "train_jsonl": str(args.train_jsonl),
            "labels_used_for_representation_training": contrastive,
            "ddp_world_size": world_size,
            "train_attn_mask_ratio": args.train_attn_mask_ratio,
            "git_commit_hash": commit_hash,
            "history": train_metrics_history,
        }
        (args.out_dir / "ssae_train_metrics.json").write_text(
            json.dumps(train_metrics, indent=2)
        )
        val_metrics = {
            "best_val_recon_ce": best_val_recon_ce,
            "last_val_recon_ce": last_val.get("val_recon_ce"),
            "last_val_l1_mean": last_val.get("val_l1_mean"),
            "history": val_metrics_history,
        }
        (args.out_dir / "ssae_val_metrics.json").write_text(
            json.dumps(val_metrics, indent=2)
        )
        logger.info(f"Saved final model + metrics to {args.out_dir}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
