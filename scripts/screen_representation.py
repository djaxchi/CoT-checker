#!/usr/bin/env python3
"""Cheap triage: is a representation worth a full grid run?

A grid cell costs minutes to hours (hyperparameter search, three seeds, the full
513,810-step split, first-error calibration over four ProcessBench subsets).
Deciding whether a new representation is promising should not cost that. This
trains one linear probe on a subsample and reports three numbers in well under a
minute.

**What it reports, and why that choice is not arbitrary.** The screen was
calibrated against 31 fully-evaluated cells, asking which cheap number predicts
the expensive headline (F1_PB at calib-20):

    in-domain PRM800K AUROC      Spearman 0.835   (0.712 on dense cells alone)
    ProcessBench step AUROC      Spearman 0.934   (0.895 on dense cells alone)

So the screen leads on ProcessBench step-level AUROC, not the in-domain number.
The in-domain number is the intuitive one and it is the worse predictor: it ranked
`step_mean x mlp:h1024` first of 31 while the full metric put it thirteenth,
because a representation can fit PRM800K well and transfer poorly. Step AUROC is
also threshold-free, so it is immune to the score saturation that broke calib-20
on the wide representations.

`signal_share` is reported alongside: the fraction of the representation's
variance explained by the class means. It needs no training at all and is the
quantity the bottleneck work is trying to raise, so it says *why* a
representation screens well or badly rather than only *that* it does.

A screen is a filter, not a verdict. It uses one seed, no hyperparameter search
and a subsample, so it will not resolve differences of a point or two. Use it to
decide what deserves the grid, and let the grid decide what is true.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.harness.bottleneck import signal_share  # noqa: E402


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(np.int64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Ties must share the average rank. Without this an all-tied score vector
    # returns 1.0, and saturated probes -- the ones that already broke calib-20 --
    # produce exactly that pattern.
    ss = np.asarray(s)[order]
    i = 0
    while i < len(ss):
        j = i + 1
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_probe(x, y, epochs: int, lr: float, batch: int, device, seed: int = 0):
    torch.manual_seed(seed)
    n, d = x.shape
    w = torch.nn.Linear(d, 1).to(device)
    opt = torch.optim.AdamW(w.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    for _ in range(epochs):
        order = rng.permutation(n)          # one shuffle per epoch, not per batch
        for i in range(0, n, batch):
            idx = torch.from_numpy(order[i:i + batch]).to(device)
            loss = F.binary_cross_entropy_with_logits(
                w(xt.index_select(0, idx)).squeeze(-1), yt.index_select(0, idx))
            opt.zero_grad(); loss.backward(); opt.step()
    return w


@torch.no_grad()
def score(w, x, device, batch=4096) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float32)
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        out[i:i + batch] = torch.sigmoid(w(xb).squeeze(-1)).float().cpu().numpy()
    return out


def screen(x_tr, y_tr, x_val, y_val, pb, device, epochs=8, lr=1e-3,
           batch=1024, seed=0) -> dict:
    """pb: list of (x, step_is_first_error) per ProcessBench subset."""
    t0 = time.perf_counter()
    share = signal_share(torch.as_tensor(np.asarray(x_tr[:20000], dtype=np.float32)),
                         torch.as_tensor(np.asarray(y_tr[:20000])))
    w = fit_probe(x_tr, y_tr, epochs, lr, batch, device, seed)
    res = {
        "signal_share": share,
        "in_domain_auroc": auroc(y_val, score(w, x_val, device)),
        "pb_step_auroc": float(np.mean([auroc(yy, score(w, xx, device)) for xx, yy in pb])),
        "n_train": int(len(x_tr)), "dim": int(x_tr.shape[1]),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, type=Path, nargs="+",
                   help="One .npz per representation with x_train,y_train,x_val,"
                        "y_val and pb_x_<sub>/pb_y_<sub> arrays.")
    p.add_argument("--n_train", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for f in args.npz:
        z = np.load(f)
        n = min(args.n_train, len(z["y_train"]))
        subs = sorted({k[5:] for k in z.files if k.startswith("pb_x_")})
        pb = [(z[f"pb_x_{s}"], z[f"pb_y_{s}"]) for s in subs]
        r = screen(z["x_train"][:n], z["y_train"][:n], z["x_val"], z["y_val"],
                   pb, device, args.epochs, args.lr)
        r["name"] = f.stem
        rows.append(r)
        print(f"[screen] {r['name']:<26} dim {r['dim']:>7}  "
              f"PB step AUROC {r['pb_step_auroc']:.4f}  "
              f"in-domain {r['in_domain_auroc']:.4f}  "
              f"signal {r['signal_share']:.5f}  ({r['seconds']}s)", flush=True)

    rows.sort(key=lambda r: -r["pb_step_auroc"])
    print(f"\n{'representation':<28}{'PB step AUROC':>14}{'in-domain':>11}{'signal share':>14}")
    for r in rows:
        print(f"{r['name']:<28}{r['pb_step_auroc']:>14.4f}{r['in_domain_auroc']:>11.4f}"
              f"{r['signal_share']:>14.5f}")
    print("\nRanked by ProcessBench step AUROC, which predicted the full metric at "
          "Spearman 0.934 over 31 evaluated cells. Differences under ~0.01 are "
          "not resolvable at this budget.")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"[screen] wrote {args.out}")


if __name__ == "__main__":
    main()
