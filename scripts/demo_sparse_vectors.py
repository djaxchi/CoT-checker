#!/usr/bin/env python3
"""Demo: encode a reasoning step and display its sparse vector.

Usage — random weights (untrained):
    python scripts/demo_sparse_vectors.py

Usage — pretrained checkpoint (downloads automatically from HuggingFace):
    python scripts/demo_sparse_vectors.py --checkpoint gsm8k-385k_Qwen2.5-0.5b_spar-10.pt

Usage — pretrained checkpoint from local file:
    python scripts/demo_sparse_vectors.py --checkpoint /path/to/ckpt.pt

Usage — custom context/step:
    python scripts/demo_sparse_vectors.py \\
        --checkpoint gsm8k-385k_Qwen2.5-0.5b_spar-10.pt \\
        --context "Let x = 5." --step "So x + 1 = 6."

When --checkpoint is given, model/sparsity are read from the checkpoint and
--model/--sparsity flags are ignored.
"""

import argparse
import sys
from pathlib import Path

import torch

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from src.saes.ssae import SSAE

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Visualise SSAE sparse vectors")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path or filename of a pretrained .pt checkpoint. "
        "If only a filename is given it is downloaded from "
        "Miaow-Lab/SSAE-Checkpoints on HuggingFace.",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model ID (ignored when --checkpoint is set)",
    )
    p.add_argument(
        "--sparsity",
        type=int,
        default=4,
        help="Sparsity expansion factor (ignored when --checkpoint is set)",
    )
    p.add_argument(
        "--context",
        default=("A store sells 7 apples for $21. We want to find the cost of one apple."),
    )
    p.add_argument("--step", default="Divide $21 by 7 to get the price per apple: 21 / 7 = 3.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--top-k", type=int, default=20, help="Show the top-k most active features")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def print_sparse_stats(vec: torch.Tensor, label: str = "") -> None:
    n = vec.numel()
    nonzero = (vec > 0).sum().item()
    sparsity = 1.0 - nonzero / n
    print(f"\n{'─' * 60}")
    if label:
        print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  Dimensionality : {n}")
    print(f"  Non-zero dims  : {nonzero}  ({100 * nonzero / n:.1f}%)")
    print(f"  Sparsity       : {sparsity:.4f}")
    print(f"  Max activation : {vec.max().item():.6f}")
    if nonzero:
        print(f"  Mean (non-zero): {vec[vec > 0].mean().item():.6f}")
    else:
        print("  Mean (non-zero): N/A")
    print(f"  L1 norm        : {vec.abs().sum().item():.4f}")
    print(f"  L2 norm        : {vec.norm().item():.4f}")


def print_top_features(vec: torch.Tensor, k: int = 20) -> None:
    nonzero_idx = vec.nonzero(as_tuple=True)[0]
    if len(nonzero_idx) == 0:
        print("\n  (all features are zero)")
        return

    actual_k = min(k, len(nonzero_idx))
    top_vals, top_idx = torch.topk(vec, actual_k)
    print(f"\n  Top-{actual_k} active features:")
    print(f"  {'Index':>8}  {'Value':>12}")
    print(f"  {'─' * 8}  {'─' * 12}")
    max_val = top_vals[0].item() + 1e-9
    for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
        bar = "█" * int(20 * val / max_val)
        print(f"  {idx:>8}  {val:>12.6f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.checkpoint:
        # --- Load from pretrained checkpoint ---
        model = SSAE.from_checkpoint(args.checkpoint, device=str(device))
        tokenizer = model.tokenizer
        model.eval()
        label = f"Pretrained checkpoint: {Path(args.checkpoint).name}"
    else:
        # --- Random-weight baseline ---
        print(f"\nLoading tokenizer from '{args.model}' …")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        # Add <sep> token to match training convention
        if tokenizer.sep_token is None:
            tokenizer.add_special_tokens({"sep_token": "<sep>"})
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

        print(f"Loading SSAE (random weights, sparsity_factor={args.sparsity}) …")
        model = SSAE(
            tokenizer=tokenizer,
            sparsity_factor=args.sparsity,
            encoder_model_id=args.model,
            decoder_model_id=args.model,
            phase=1,
        ).to(device)
        model.eval()
        label = f"Random weights  [model={args.model}, F={args.sparsity}]"

    # --- Tokenize [context <sep> step] ---
    sep = tokenizer.sep_token  # <sep>, as used during training
    full_text = args.context + sep + args.step

    print("\nInput:")
    print(f"  Context : {args.context!r}")
    print(f"  Sep     : {sep!r}")
    print(f"  Step    : {args.step!r}")

    encoding = tokenizer(
        full_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # --- Encode → sparse vector ---
    sparse_vec = model.get_sparse_vector(input_ids, attention_mask)  # (1, n_latents)
    vec = sparse_vec[0].cpu()  # (n_latents,)

    print_sparse_stats(vec, label=label)
    print_top_features(vec, k=args.top_k)

    print(f"\n  Raw tensor shape : {tuple(vec.shape)}")
    print(f"  dtype            : {vec.dtype}")
    print()


if __name__ == "__main__":
    main()
