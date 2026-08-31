#!/usr/bin/env python3
"""Check that the screen's verdict is about the representation, not the budget.

Two scripts fitting the same probe on the same file disagreed by 0.02 on the
8,192-dim stacked representations and by 0.004 on the 4,096-dim ones. The only
difference between them was the number of training rows. A gap that grows with
dimension is the signature of an undertrained probe: at a fixed epoch count and
learning rate, a wider representation is further from convergence than a narrow
one, so the screen would be ranking how quickly a representation can be fitted
rather than how much it separates.

That matters most for exactly the comparison the search is now making. Stacking
two layers doubles the width, so "the stack wins" and "the stack is merely
evaluated at a different point on its optimisation path" are the same number
until this is run.

Sweeps epochs and learning rate for each input and prints the whole grid. The
readable outcome is not the best cell, it is whether the ORDER of the
representations is the same in every cell. If it is, the screen's budget is
irrelevant to its verdict. If it is not, the screen is measuring optimisation
speed and every previous ranking has to be re-read at convergence.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_representation import auroc, fit_probe, score, standardize  # noqa: E402


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", nargs="+", required=True, type=Path)
    p.add_argument("--epochs", nargs="+", type=int, default=[8, 25, 60])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 1e-2])
    p.add_argument("--n_train", type=int, default=50000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cells, names = {}, [f.stem for f in args.npz]
    for f in args.npz:
        z = np.load(f)
        n = min(args.n_train, len(z["y_train"]))
        xtr, ytr = z["x_train"][:n], z["y_train"][:n]
        mu, sd = standardize(xtr)
        zs = lambda a: (np.asarray(a, dtype=np.float32) - mu) / sd     # noqa: E731
        subs = sorted({k[5:] for k in z.files if k.startswith("pb_x_")})
        pbx = [(zs(z[f"pb_x_{s}"]), z[f"pb_y_{s}"]) for s in subs]
        xz = zs(xtr)
        for ep, lr in product(args.epochs, args.lrs):
            w = fit_probe(xz, ytr, ep, lr, args.batch, dev)
            cells[(f.stem, ep, lr)] = float(np.mean(
                [auroc(y, score(w, x, dev)) for x, y in pbx]))
        print(f"[sweep] {f.stem} dim {xtr.shape[1]} done", flush=True)

    cols = list(product(args.epochs, args.lrs))
    head = "".join(f"{f'e{e} lr{l:g}':>14}" for e, l in cols)
    print(f"\n{'representation':<26}{head}")
    for nm in names:
        print(f"{nm:<26}" + "".join(f"{cells[(nm, e, l)]:>14.4f}" for e, l in cols))

    base = np.array([cells[(nm, *cols[0])] for nm in names])
    print(f"\nrank agreement with the screen's own budget "
          f"(epochs {cols[0][0]}, lr {cols[0][1]:g}):")
    for c in cols[1:]:
        v = np.array([cells[(nm, *c)] for nm in names])
        print(f"  epochs {c[0]:>3} lr {c[1]:<8g} Spearman {spearman(base, v):>6.3f}   "
              f"best: {names[int(np.argmax(v))]}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {f"{k[0]}|e{k[1]}|lr{k[2]}": v for k, v in cells.items()}, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
