#!/usr/bin/env python3
"""Build a surface-feature baseline: what can be predicted without activations.

The grid has never had one, and it needs one. Removing the length-correlated
component of `step_mean` drops the between-class variance by 56% (signal share
0.00704 -> 0.00326 while total variance falls only 5.4%), which means roughly
half of what the probe discriminates on is length-correlated. Removing it also
costs up to 0.063 AUROC, so it is real signal rather than noise — but that raises
the obvious question the leaderboard cannot currently answer:

    how much of the leaderboard is reading the reasoning, and how much is
    reading how long the step is?

If step length alone reaches a large fraction of `step_mean`'s transfer score,
the interesting range of the whole grid shrinks accordingly, and every claim about
representations has to be stated net of it. That is a result either way, and it is
cheap.

Emits `length` (log tokens alone), `length_poly` (with squared and raw terms, so a
non-linear length response is not mistaken for its absence), and `augment`, which
concatenates the length features onto a representation.

`augment` is what makes the baseline answer the question rather than merely raise
it. Comparing three numbers -- surface alone, representation alone, and
representation plus surface -- separates the cases:

    rep+surface ~= surface          the activations add nothing over length
    rep ~= rep+surface >> surface   the representation already contains length,
                                    and more besides
    rep+surface > both              they carry different things

Without the augmented cell, a representation scoring above the surface baseline
could still be doing nothing except reading length more accurately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def feats(lengths: np.ndarray, mode: str) -> np.ndarray:
    lg = np.log(np.maximum(lengths, 1.0)).astype(np.float32)
    if mode == "length":
        return lg[:, None]
    return np.stack([lg, lg ** 2, lengths.astype(np.float32)], 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, type=Path,
                   help="Any pooling npz; only its length and label arrays are used.")
    p.add_argument("--mode", choices=["length", "length_poly", "augment"],
                   default="length")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    z = dict(np.load(args.npz))
    if "len_train" not in z:
        raise SystemExit(f"{args.npz} has no length arrays")
    subs = sorted({k[5:] for k in z if k.startswith("pb_x_")})
    out = {"y_train": z["y_train"], "y_val": z["y_val"]}
    if args.mode == "augment":
        def cat(x, lg):
            return np.concatenate([x.astype(np.float32), feats(lg, "length_poly")], 1)
        out["x_train"] = cat(z["x_train"], z["len_train"])
        out["x_val"] = cat(z["x_val"], z["len_val"])
        for s in subs:
            out[f"pb_x_{s}"] = cat(z[f"pb_x_{s}"], z[f"pb_len_{s}"])
    else:
        out["x_train"] = feats(z["len_train"], args.mode)
        out["x_val"] = feats(z["len_val"], args.mode)
        for s in subs:
            out[f"pb_x_{s}"] = feats(z[f"pb_len_{s}"], args.mode)
    for s in subs:
        out[f"pb_y_{s}"] = z[f"pb_y_{s}"]
        out[f"pb_len_{s}"] = z[f"pb_len_{s}"]
    # keep the lengths so the output can still be scored within length strata
    out["len_train"], out["len_val"] = z["len_train"], z["len_val"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)

    # the plainest possible statement of the effect, before any probe is fitted
    lt, yt = z["len_train"], z["y_train"]
    pl = np.concatenate([z[f"pb_len_{s}"] for s in subs])
    py = np.concatenate([z[f"pb_y_{s}"] for s in subs])
    print(f"[surface] {args.mode}: dim {out['x_train'].shape[1]}")
    print(f"  train  mean length  correct {lt[yt==0].mean():.1f}  "
          f"incorrect {lt[yt==1].mean():.1f}")
    print(f"  procbench mean length  not-first-error {pl[py==0].mean():.1f}  "
          f"first-error {pl[py==1].mean():.1f}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
