#!/usr/bin/env python3
"""Is the geometry block acting as a suppressor variable?

The ablation left a puzzle. The 20 geometry features are worth +0.0614 on top of
length-free content, and any ONE of about eight of them recovers most of that
(`cone_tightness_ratio` +0.0458, `cone_cos_mean` +0.0457). Yet the whole block
scores 0.5182 alone, and 0.4675 inside length strata, which is below chance.

A feature that carries nothing by itself and a lot in combination has a name in
regression: a suppressor. It correlates weakly or not at all with the outcome,
and it earns its place by explaining variance in the OTHER predictors that is
unrelated to the outcome, so that what remains of them lines up better. The
prediction is sharp and falsifiable:

    corr(feature, label)          near zero
    corr(feature, content score)  clearly non-zero
    partial corr(feature, label | content score)  clearly non-zero, and larger
                                                  in magnitude than the raw one

If instead the raw correlation is large, the feature is ordinary signal and the
low standalone AUROC needs a different explanation. If the partial correlation is
also near zero, the gain is not coming from these features at all and the
add-one-in numbers are an artifact.

The alternative worth ruling out at the same time: cone tightness could simply be
a nonlinear stand-in for step length, since more tokens spread a cone. That is
already argued against by the block scoring 0.5182 where length alone scores
0.7039, but the correlation with log length is reported here so the claim rests
on a number rather than an inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from screen_representation import auroc, standardize  # noqa: E402
from ridge_screen import ridge_path  # noqa: E402
from src.harness.geom import GEOM_NAMES, N_GEOM  # noqa: E402


def partial_corr(a, b, c) -> float:
    """corr(a, b) with c linearly removed from both."""
    a, b, c = (np.asarray(v, dtype=np.float64) for v in (a, b, c))
    d = np.stack([np.ones_like(c), c], 1)
    ra = a - d @ np.linalg.lstsq(d, a, rcond=None)[0]
    rb = b - d @ np.linalg.lstsq(d, b, rcond=None)[0]
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, type=Path)
    p.add_argument("--n_train", type=int, default=50000)
    p.add_argument("--lambdas", nargs="+", type=float, default=[1e3, 1e4, 1e5])
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    z = np.load(args.npz)
    total = z["x_train"].shape[1]
    d = total - N_GEOM
    subs = sorted({k[5:] for k in z.files if k.startswith("pb_x_")})

    # a probe on the length-free CONTENT only, which is what the geometry is
    # being asked to correct
    n = min(args.n_train, len(z["y_train"]))
    xtr = np.asarray(z["x_train"][:n, :d], np.float32)
    ytr = z["y_train"][:n]
    mu, sd = standardize(xtr)
    f = lambda a: (np.asarray(a, np.float64) - mu) / sd            # noqa: E731
    ws = ridge_path(f(xtr), ytr, args.lambdas)
    xv, yv = f(z["x_val"][:, :d]), z["y_val"]
    lam = max(args.lambdas, key=lambda L: auroc(yv, xv @ ws[L]))
    w = ws[lam]

    # evaluated on ProcessBench, since that is where the gain appears
    g = np.concatenate([z[f"pb_x_{s}"][:, d:] for s in subs])
    y = np.concatenate([z[f"pb_y_{s}"] for s in subs])
    score = np.concatenate([f(z[f"pb_x_{s}"][:, :d]) @ w for s in subs])
    lg = np.log(np.maximum(np.concatenate([z[f"pb_len_{s}"] for s in subs]), 1.0))

    print(f"content-only probe on ProcessBench: AUROC {auroc(y, score):.4f} "
          f"(penalty {lam:.0e}, chosen on validation)\n")
    print(f"{'feature':<24}{'corr(f,y)':>11}{'corr(f,score)':>15}"
          f"{'partial(f,y|score)':>20}{'corr(f,log len)':>17}")
    rows = []
    for j, name in enumerate(GEOM_NAMES):
        fj = g[:, j].astype(np.float64)
        if fj.std() < 1e-12:
            continue
        r_y = float(np.corrcoef(fj, y)[0, 1])
        r_s = float(np.corrcoef(fj, score)[0, 1])
        r_p = partial_corr(fj, y, score)
        r_l = float(np.corrcoef(fj, lg)[0, 1])
        rows.append({"feature": name, "corr_y": r_y, "corr_score": r_s,
                     "partial_y_given_score": r_p, "corr_loglen": r_l})
        print(f"{name:<24}{r_y:>+11.4f}{r_s:>+15.4f}{r_p:>+20.4f}{r_l:>+17.4f}")

    sup = [r for r in rows if abs(r["corr_y"]) < 0.10
           and abs(r["partial_y_given_score"]) > 2 * abs(r["corr_y"])
           and abs(r["corr_score"]) > 0.20]
    print(f"\nsuppressor pattern (weak with the label, strong with the probe's "
          f"score, stronger once the score is removed):")
    for r in sorted(sup, key=lambda r: -abs(r["partial_y_given_score"]))[:6]:
        print(f"  {r['feature']:<24} corr(f,y) {r['corr_y']:+.4f} -> "
              f"partial {r['partial_y_given_score']:+.4f}")
    if not sup:
        print("  none: the gain is ordinary signal, not suppression")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
