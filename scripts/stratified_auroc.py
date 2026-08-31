#!/usr/bin/env python3
"""Score representations within length strata, where step length cannot help.

Step length alone scores 0.7039 on ProcessBench while the best representation
scores 0.7700, so most of the transfer benchmark is answerable from a token
count. That number is not a property of any representation, it is a property of
how ProcessBench is built: the first erroneous step runs 118.6 tokens against
79.7 for the rest, and PRM800K steps average 38.8, so the probe is trained short
and tested long.

Ranking representations on a score that a token count mostly reproduces tells you
very little. This computes the same AUROC inside equal-count length bins and
averages the bins by their pair counts. Inside a bin the steps are the same
length, so length carries no information and whatever separation remains is the
representation's own.

Two checks make the result readable rather than merely smaller:
  - the length baseline is run through the identical procedure and printed last.
    It should collapse toward 0.5; if it sits above, the bins are too wide and
    every row above it is still partly a length score.
  - both the plain and the stratified number are printed for every input, so the
    drop is attributable per representation instead of assumed uniform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_representation import auroc, fit_probe, score, standardize  # noqa: E402


def stratified_auroc(y, s, lengths, n_bins: int):
    """AUROC inside equal-count length bins, weighted by comparable pairs.

    Weighting by npos*nneg rather than by bin size is what makes this the same
    quantity as the plain AUROC restricted to same-length pairs: a bin holding
    one positive contributes proportionally to the comparisons it supports, not
    to the rows it holds.
    """
    order = np.argsort(lengths, kind="mergesort")
    bins = np.array_split(order, n_bins)
    num = den = 0.0
    for b in bins:
        yb = y[b]
        npos, nneg = int((yb == 1).sum()), int((yb == 0).sum())
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += w * auroc(yb, s[b])
        den += w
    return float(num / den) if den else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", nargs="+", required=True, type=Path)
    p.add_argument("--n_bins", type=int, default=50,
                   help="Equal-count length bins. Coarse bins leave length usable: "
                        "on ProcessBench-shaped data the length control still "
                        "scores 0.60 at 5 bins and 0.54 at 10, and only reaches "
                        "0.50 near 50. Read the printed control before the rows.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--n_train", type=int, default=50000,
                   help="Match screen_representation.py exactly. The two scripts "
                        "fit the same probe, so any difference in rows, epochs or "
                        "lr makes their PB columns silently incomparable.")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    print(f"{'representation':<26} {'PB plain':>9} {'PB within-len':>14} "
          f"{'cost':>7} {'dim':>7}")
    for path in args.npz:
        z = np.load(path)
        subs = sorted({k[5:] for k in z.files if k.startswith("pb_x_")})
        if f"pb_len_{subs[0]}" not in z.files:
            print(f"{path.stem:<26} no length arrays, skipped")
            continue
        n = min(args.n_train, len(z["y_train"]))
        xtr, ytr = z["x_train"][:n], z["y_train"][:n]
        mu, sd = standardize(xtr)
        zs = lambda a: (np.asarray(a, dtype=np.float32) - mu) / sd   # noqa: E731
        w = fit_probe(zs(xtr), ytr, args.epochs, args.lr, args.batch, dev)
        plain, strat = [], []
        for s in subs:
            sc = score(w, zs(z[f"pb_x_{s}"]), dev)
            y, ln = z[f"pb_y_{s}"], z[f"pb_len_{s}"]
            plain.append(auroc(y, sc))
            strat.append(stratified_auroc(y, sc, ln, args.n_bins))
        r = {"name": path.stem, "dim": int(z["x_train"].shape[1]),
             "pb_plain": float(np.mean(plain)),
             "pb_within_length": float(np.mean(strat)),
             "per_subset": dict(zip(subs, map(float, strat)))}
        rows.append(r)
        print(f"{r['name']:<26} {r['pb_plain']:>9.4f} {r['pb_within_length']:>14.4f} "
              f"{r['pb_plain'] - r['pb_within_length']:>7.4f} {r['dim']:>7d}")

    # the control: length scored by the identical procedure must collapse to ~0.5
    z = np.load(args.npz[0])
    subs = sorted({k[5:] for k in z.files if k.startswith("pb_x_")})
    ctl = [stratified_auroc(z[f"pb_y_{s}"], z[f"pb_len_{s}"].astype(np.float64),
                            z[f"pb_len_{s}"], args.n_bins) for s in subs]
    ctlp = [auroc(z[f"pb_y_{s}"], z[f"pb_len_{s}"].astype(np.float64)) for s in subs]
    print(f"{'[control] length itself':<26} {np.mean(ctlp):>9.4f} "
          f"{np.mean(ctl):>14.4f} {np.mean(ctlp) - np.mean(ctl):>7.4f}")
    rows.append({"name": "[control] length itself", "dim": 1,
                 "pb_plain": float(np.mean(ctlp)),
                 "pb_within_length": float(np.mean(ctl))})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
