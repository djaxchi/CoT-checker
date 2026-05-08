#!/usr/bin/env python3
"""Derive all representation .npz files needed for PTB probe evaluation.

Given pre-encoded dense h_k vectors (already in .npz files), this script
produces one .npz per representation per dataset -- all in the exact field
format expected by eval_predictive_transition_probes.py.

Representations produced:
  ptb_<name>     h_k -> PTB encoder -> z_k (896)  for each --ptb-checkpoint
  dense_delta_h  delta_h_k = h_{k+1} - h_k        (896, steps with predecessor only)
  dense_concat   [h_k ; delta_h_k]                 (1792, same mask as delta)
  random_bln     random Linear+ReLU z_k            (896, same arch as PTB, random init)

The dense h_k baseline is the input file itself -- not re-generated.
The SSAE z_k baseline already exists from previous runs -- not re-generated.

Usage:
    python scripts/extract_predictive_transition_latents.py \\
        --ptb-checkpoints ptb_no_l1/best.pt:no_l1 ptb_fixed/best.pt:fixed_lambda ... \\
        --ms-dense-eval  dense_eval_held_out.npz \\
        --pb-dense       processbench_dense_gsm8k.npz \\
        --ms-dense-train dense_train_full.npz \\
        --output-dir     ptb_representations/ \\
        --device         cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.saes.ptb import PredictiveTransitionBottleneck


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_ptb(
    ptb: PredictiveTransitionBottleneck,
    h: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    ptb.eval()
    out = []
    h_t = torch.from_numpy(h)
    for i in range(0, len(h), batch_size):
        x = h_t[i:i + batch_size].to(device)
        z = ptb.encode(x)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def random_bottleneck_encode(h: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random Linear(d,d)+ReLU projection -- same arch as PTB, untrained weights."""
    rng = torch.Generator().manual_seed(seed)
    d = h.shape[1]
    W = torch.empty(d, d)
    torch.nn.init.xavier_uniform_(W, generator=rng)
    b = torch.zeros(d)
    h_t = torch.from_numpy(h)
    z = F.relu(h_t @ W.T + b)
    return z.numpy().astype(np.float32)


def compute_delta_representations(
    h: np.ndarray,
    solution_ids: np.ndarray,
    step_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute delta_h for steps that have a predecessor in the same solution.

    Returns (delta_h, keep_mask) where keep_mask is (N,) bool.
    Steps at position 0 within a solution have no predecessor and are excluded.
    """
    keep  = np.zeros(len(h), dtype=bool)
    delta = np.zeros_like(h)

    for i in range(1, len(h)):
        if (solution_ids[i] == solution_ids[i - 1]
                and step_positions[i] == step_positions[i - 1] + 1):
            delta[i] = h[i].astype(np.float32) - h[i - 1].astype(np.float32)
            keep[i]  = True

    return delta[keep], keep


def _save(
    path: Path,
    latents: np.ndarray,
    source_npz: np.lib.npyio.NpzFile,
    keep_mask: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"latents": latents.astype(np.float32)}
    for key in source_npz.files:
        if key == "latents":
            continue
        arr = source_npz[key]
        if keep_mask is not None and hasattr(arr, "shape") and arr.shape[0] == keep_mask.shape[0]:
            arr = arr[keep_mask]
        kwargs[key] = arr
    np.savez_compressed(path, **kwargs)
    print(f"  Saved {latents.shape} -> {path.name}")


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    label: str,
    dense_npz_path: str,
    ptb_variants: list[tuple[PredictiveTransitionBottleneck, str]],
    out_dir: Path,
    batch_size: int,
    device: str,
) -> None:
    """Produce all representation files for one dataset split."""
    print(f"\n[{label}] Loading {dense_npz_path} ...")
    if not Path(dense_npz_path).exists():
        print(f"  SKIP: file not found: {dense_npz_path}")
        return
    dn = np.load(dense_npz_path)
    h  = dn["latents"].astype(np.float32)
    N  = len(h)
    print(f"  {N:,} steps  dim={h.shape[1]}")

    # PTB variants
    for ptb, name in ptb_variants:
        print(f"  Encoding PTB z_k [{name}] ...")
        ptb_z = apply_ptb(ptb, h, batch_size, device)
        _save(out_dir / f"ptb_{name}_{label}.npz", ptb_z, dn)

    # Random bottleneck (untrained, same arch)
    print(f"  Random bottleneck ...")
    rand_z = random_bottleneck_encode(h, seed=0)
    _save(out_dir / f"random_bln_{label}.npz", rand_z, dn)

    # Delta representations (need solution_ids + step_positions)
    if "solution_ids" in dn.files and "step_positions" in dn.files:
        sol_ids  = dn["solution_ids"]
        step_pos = dn["step_positions"]
        print(f"  Computing delta_h_k ...")
        delta_h, keep_mask = compute_delta_representations(h, sol_ids, step_pos)
        print(f"  {keep_mask.sum():,} steps with predecessor  "
              f"(dropped {(~keep_mask).sum():,} first-of-solution steps)")
        _save(out_dir / f"dense_delta_{label}.npz",  delta_h,                                        dn, keep_mask)
        _save(out_dir / f"dense_concat_{label}.npz", np.concatenate([h[keep_mask], delta_h], axis=1), dn, keep_mask)
    else:
        print(f"  WARNING: no solution_ids in {dense_npz_path}; skipping delta/concat reps.")
        print(f"  Run add_ms_solution_ids.py first to patch those fields.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Derive PTB + baseline representation .npz files")

    p.add_argument(
        "--ptb-checkpoints", nargs="*", default=[],
        metavar="PATH:NAME",
        help=(
            "One or more PTB checkpoints as path:name pairs, e.g. "
            "ptb_no_l1/best.pt:no_l1 ptb_fixed/best.pt:fixed_lambda"
        ),
    )

    # Math-Shepherd splits
    p.add_argument("--ms-dense-eval",   required=True, help="dense_eval_held_out.npz")
    p.add_argument("--pb-dense",        default=None,  help="processbench_dense_gsm8k.npz (optional)")
    p.add_argument("--ms-dense-train",  required=True, help="dense_train_full.npz")
    p.add_argument(
        "--datasets", nargs="+",
        default=["ms_eval", "pb", "ms_train"],
        choices=["ms_eval", "pb", "ms_train"],
        help="Which dataset splits to process (default: all three)",
    )

    p.add_argument("--output-dir",      required=True)
    p.add_argument("--batch-size",      type=int, default=4096)
    p.add_argument("--device",          default="cpu", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse PTB checkpoints
    ptb_variants: list[tuple[PredictiveTransitionBottleneck, str]] = []
    for spec in args.ptb_checkpoints:
        if ":" not in spec:
            raise ValueError(f"--ptb-checkpoints expects PATH:NAME format, got: {spec!r}")
        path_str, name = spec.rsplit(":", 1)
        path = Path(path_str)
        if not path.exists():
            print(f"  WARNING: checkpoint not found, skipping: {path}")
            continue
        print(f"Loading PTB [{name}] from {path} ...")
        ptb = PredictiveTransitionBottleneck.from_checkpoint(path, device=args.device)
        print(f"  d={ptb.d}  k={ptb.k}")
        ptb_variants.append((ptb, name))

    if not ptb_variants:
        print("WARNING: no valid PTB checkpoints provided. "
              "Only random_bln and delta reps will be generated.")

    all_datasets = [
        ("ms_eval",  args.ms_dense_eval),
        ("pb",       args.pb_dense),
        ("ms_train", args.ms_dense_train),
    ]
    datasets = [
        (label, path) for label, path in all_datasets
        if label in args.datasets and path is not None
    ]
    if "pb" in args.datasets and args.pb_dense is None:
        print("NOTE: --pb-dense not provided; skipping pb dataset.")
    for label, dense_path in datasets:
        process_dataset(
            label        = label,
            dense_npz_path = dense_path,
            ptb_variants = ptb_variants,
            out_dir      = out_dir,
            batch_size   = args.batch_size,
            device       = args.device,
        )

    print("\n=== All representations written ===")
    for f in sorted(out_dir.glob("*.npz")):
        d = np.load(f)
        has_labels = "correctness" in d.files or "step_labels" in d.files
        print(f"  {f.name:55s}  shape={d['latents'].shape}  labels={has_labels}")


if __name__ == "__main__":
    main()
