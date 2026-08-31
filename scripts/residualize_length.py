#!/usr/bin/env python3
"""Regress step length out of a representation, or add it in, and re-screen.

A measured fact that nothing in the grid has used: PRM800K steps average 38.8
tokens while ProcessBench steps run 56 to 94. The probe is trained on the short
domain and evaluated on the long one. If the representation encodes length, the
probe learns a length-dependent decision boundary that misfires on the longer
domain, and the transfer number pays for it.

Two directions, because the pair is what makes the test conclusive:

  residual   fit a linear map from log(length) to each position on TRAIN, then
             subtract its prediction everywhere. Statistics come from train only
             and are applied unchanged to ProcessBench; refitting per split would
             remove whatever genuine length effect exists in each domain
             separately and prove nothing.
  withlen    concat[representation, log(length)] -- the opposite control.

If `withlen` helps in domain and hurts transfer while `residual` does the
reverse, length is a transfer confound and removing it is a real gain. If neither
moves anything, length is not what the representation is carrying and the
hypothesis is dead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def fit_length_map(x: np.ndarray, lengths: np.ndarray):
    """Least-squares map from [1, log len] to each position, fitted on train."""
    a = np.stack([np.ones_like(lengths), np.log(np.maximum(lengths, 1.0))], 1)
    coef, *_ = np.linalg.lstsq(a.astype(np.float64), x.astype(np.float64), rcond=None)
    return coef.astype(np.float32)


def apply_residual(x, lengths, coef):
    a = np.stack([np.ones_like(lengths), np.log(np.maximum(lengths, 1.0))], 1)
    return (x.astype(np.float32) - a.astype(np.float32) @ coef)


def apply_withlen(x, lengths):
    return np.concatenate([x.astype(np.float32),
                           np.log(np.maximum(lengths, 1.0))[:, None].astype(np.float32)], 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, type=Path)
    p.add_argument("--mode", choices=["residual", "withlen"], required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    z = dict(np.load(args.npz))
    if "len_train" not in z:
        raise SystemExit(f"{args.npz} has no length arrays; re-derive with the "
                         f"current screen_poolings.py")
    subs = sorted({k[5:] for k in z if k.startswith("pb_x_")})
    coef = fit_length_map(z["x_train"], z["len_train"]) if args.mode == "residual" else None

    out = {"y_train": z["y_train"], "y_val": z["y_val"]}
    if args.mode == "residual":
        out["x_train"] = apply_residual(z["x_train"], z["len_train"], coef)
        out["x_val"] = apply_residual(z["x_val"], z["len_val"], coef)
        for s in subs:
            out[f"pb_x_{s}"] = apply_residual(z[f"pb_x_{s}"], z[f"pb_len_{s}"], coef)
    else:
        out["x_train"] = apply_withlen(z["x_train"], z["len_train"])
        out["x_val"] = apply_withlen(z["x_val"], z["len_val"])
        for s in subs:
            out[f"pb_x_{s}"] = apply_withlen(z[f"pb_x_{s}"], z[f"pb_len_{s}"])
    for s in subs:
        out[f"pb_y_{s}"] = z[f"pb_y_{s}"]

    # How much of the representation was length, and how far apart the domains are
    var_removed = float("nan")
    if args.mode == "residual":
        v0 = z["x_train"].astype(np.float64).var(0).sum()
        v1 = out["x_train"].astype(np.float64).var(0).sum()
        var_removed = float(1 - v1 / max(v0, 1e-12))
    tl = float(np.mean(z["len_train"]))
    pl = float(np.mean(np.concatenate([z[f"pb_len_{s}"] for s in subs])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"[len] {args.npz.stem} -> {args.mode}: dim {out['x_train'].shape[1]}, "
          f"variance explained by length {100*var_removed:.2f}%, "
          f"mean length train {tl:.1f} vs ProcessBench {pl:.1f}  -> {args.out}")


if __name__ == "__main__":
    main()
