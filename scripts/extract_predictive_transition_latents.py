#!/usr/bin/env python3
"""Derive all representation .npz files needed for PTB probe evaluation.

Given pre-encoded dense h_k vectors (already in .npz files), this script
produces one .npz per representation per dataset — all in the exact field
format expected by experiment_linear_probe.py and eval_processbench.py.

Representations produced for each input dataset:
  ptb_z          h_k → PTB encoder → z_k (896)
  dense_delta_h  Δh_k = h_{k+1} - h_k    (896, only steps with predecessor)
  dense_concat   [h_k ; Δh_k]             (1792, same keep_mask as delta)
  random_bln     random Linear+ReLU z_k   (896, same arch as PTB, random init)

The dense h_k baseline is the input file itself — not re-generated.
The SSAE z_k baseline already exists from previous runs — not re-generated.

Usage:
    python scripts/extract_predictive_transition_latents.py \\
        --ptb-checkpoint $STORE/results/checkpoints/ptb_c1/best.pt \\
        --ms-dense-eval  $SCRATCH/cot-checker/probe_data/dense_eval_held_out.npz \\
        --ms-trans-eval  $SCRATCH/cot-checker/probe_data/transition_ms_eval.npz \\
        --pb-dense       $SCRATCH/cot-checker/processbench/processbench_dense_gsm8k.npz \\
        --pb-trans       $SCRATCH/cot-checker/probe_data/transition_pb.npz \\
        --ms-dense-train $SCRATCH/cot-checker/probe_data/dense_train_full.npz \\
        --ms-trans-train $SCRATCH/cot-checker/probe_data/transition_train_positive.npz \\
        --output-dir     $SCRATCH/cot-checker/probe_data/ptb_representations \\
        --device cpu
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
    h: np.ndarray,       # (N, d) float32
    batch_size: int,
    device: str,
) -> np.ndarray:         # (N, d) float32
    ptb.eval()
    out = []
    h_t = torch.from_numpy(h)
    for i in range(0, len(h), batch_size):
        x = h_t[i:i + batch_size].to(device)
        z = ptb.encode(x)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def random_bottleneck_encode(h: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random Linear(d,d)+ReLU projection — same arch as PTB, untrained weights."""
    rng = torch.Generator().manual_seed(seed)
    d = h.shape[1]
    W = torch.empty(d, d)
    torch.nn.init.xavier_uniform_(W, generator=rng)
    b = torch.zeros(d)
    h_t = torch.from_numpy(h)
    z = F.relu(h_t @ W.T + b)
    return z.numpy()


def compute_delta_representations(
    h: np.ndarray,              # (N, d)
    solution_ids: np.ndarray,   # (N,)
    step_positions: np.ndarray, # (N,)
) -> tuple[np.ndarray, np.ndarray]:
    """Compute delta_h for steps that have a predecessor in the same solution.

    Returns (delta_h, keep_mask) where keep_mask is (N,) bool.
    Steps at position 0 within a solution have no predecessor and are excluded.
    """
    keep = np.zeros(len(h), dtype=bool)
    delta = np.zeros_like(h)

    for i in range(1, len(h)):
        if (solution_ids[i] == solution_ids[i - 1]
                and step_positions[i] == step_positions[i - 1] + 1):
            delta[i] = h[i].astype(np.float32) - h[i - 1].astype(np.float32)
            keep[i] = True

    return delta[keep], keep


def _save(path: Path, latents: np.ndarray, source_npz: dict, keep_mask: np.ndarray | None = None) -> None:
    """Save latents + all metadata from source_npz, optionally filtered by keep_mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, np.ndarray] = {"latents": latents.astype(np.float32)}
    for key in source_npz.files:
        if key == "latents":
            continue
        arr = source_npz[key]
        if keep_mask is not None and arr.shape[0] == keep_mask.shape[0]:
            arr = arr[keep_mask]
        kwargs[key] = arr
    np.savez_compressed(path, **kwargs)
    print(f"  Saved {latents.shape} → {path}")


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    label: str,
    dense_npz_path: str,
    trans_npz_path: str | None,
    ptb: PredictiveTransitionBottleneck,
    out_dir: Path,
    batch_size: int,
    device: str,
) -> None:
    """Produce all four representation files for one dataset."""
    print(f"\n[{label}] Loading {dense_npz_path} ...")
    dn = np.load(dense_npz_path)
    h  = dn["latents"].astype(np.float32)
    N  = len(h)
    print(f"  {N:,} steps  dim={h.shape[1]}")

    # PTB z_k
    print(f"  Encoding PTB z_k ...")
    ptb_z = apply_ptb(ptb, h, batch_size, device)
    _save(out_dir / f"ptb_z_{label}.npz", ptb_z, dn)

    # Random bottleneck
    print(f"  Random bottleneck ...")
    rand_z = random_bottleneck_encode(h, seed=0)
    _save(out_dir / f"random_bln_{label}.npz", rand_z, dn)

    # Delta representations — require solution_ids + step_positions
    # Either from the dense .npz itself (if it has them) or from a separate trans .npz
    if "solution_ids" in dn.files and "step_positions" in dn.files:
        sol_ids  = dn["solution_ids"]
        step_pos = dn["step_positions"]
    elif trans_npz_path is not None:
        print(f"  Loading transition pairs from {trans_npz_path} for delta computation ...")
        tn = np.load(trans_npz_path)
        # The transition .npz stores h_k and h_next; we can derive delta from it directly.
        # But to align with the dense .npz correctness labels, we need keep_mask from dense.
        # Simpler: if the transition .npz has h_k + delta_h, encode h_k through PTB
        # and reuse delta_h for the delta rep. But the dense .npz and trans .npz may differ in N.
        # Safest: compute delta from dense h using problem_id + step_idx from trans .npz.
        # For now, fall through to the no-solution-ids path.
        sol_ids  = None
        step_pos = None
    else:
        sol_ids  = None
        step_pos = None

    if sol_ids is not None and step_pos is not None:
        print(f"  Computing Δh_k ...")
        delta_h, keep_mask = compute_delta_representations(h, sol_ids, step_pos)
        print(f"  {keep_mask.sum():,} steps with predecessor (dropped {(~keep_mask).sum():,})")
        _save(out_dir / f"dense_delta_{label}.npz",  delta_h,                                    dn, keep_mask)
        _save(out_dir / f"dense_concat_{label}.npz", np.concatenate([h[keep_mask], delta_h], axis=1), dn, keep_mask)
    else:
        print(f"  WARNING: no solution_ids in dense .npz and no --trans-npz for {label}; "
              f"skipping delta/concat representations.")
        print(f"  Ensure the dense .npz was created with generate_probe_data.py which saves solution_ids.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Derive PTB + baseline representation .npz files")
    p.add_argument("--ptb-checkpoint",  required=True, help="Trained PTB best.pt")
    # Math-Shepherd eval
    p.add_argument("--ms-dense-eval",   required=True, help="dense_eval_held_out.npz")
    p.add_argument("--ms-trans-eval",   default=None,  help="transition_ms_eval.npz (optional)")
    # ProcessBench
    p.add_argument("--pb-dense",        required=True, help="processbench_dense_gsm8k.npz")
    p.add_argument("--pb-trans",        default=None,  help="transition_pb.npz (optional)")
    # Math-Shepherd train (for probe training)
    p.add_argument("--ms-dense-train",  required=True, help="dense_train_full.npz")
    p.add_argument("--ms-trans-train",  default=None,  help="transition_train_positive.npz (optional)")
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--batch-size",      type=int, default=4096)
    p.add_argument("--device",          default="cpu", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PTB from {args.ptb_checkpoint} ...")
    ptb = PredictiveTransitionBottleneck.from_checkpoint(args.ptb_checkpoint, device=args.device)
    print(f"  d={ptb.d}")

    datasets = [
        ("ms_eval",   args.ms_dense_eval,  args.ms_trans_eval),
        ("pb",        args.pb_dense,        args.pb_trans),
        ("ms_train",  args.ms_dense_train,  args.ms_trans_train),
    ]
    for label, dense_path, trans_path in datasets:
        process_dataset(
            label      = label,
            dense_npz_path = dense_path,
            trans_npz_path = trans_path,
            ptb        = ptb,
            out_dir    = out_dir,
            batch_size = args.batch_size,
            device     = args.device,
        )

    print("\n=== All representations written ===")
    for f in sorted(out_dir.glob("*.npz")):
        d = np.load(f)
        print(f"  {f.name:50s}  {d['latents'].shape}  labels={'correctness' in d.files or 'step_labels' in d.files}")


if __name__ == "__main__":
    main()
