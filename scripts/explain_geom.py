#!/usr/bin/env python3
"""Which of the 20 geometry features actually carry the gain?

`lengthfree_geom` beats a dimension-matched `step_mean` by 0.026 F1 at calib-20
and by 0.043 ProcessBench AUROC under a ridge probe. The project's stated mission
is to explain the signal rather than to move the score, and right now the
explanation is a 20-dimensional shrug: the features were designed together, they
are useless on their own (0.5182), and they help only in combination.

Two ablations over the same closed-form ridge path, so neither answer depends on
an optimiser:

  add-one-in     mean_residual plus ONE geometry feature. Says what a feature is
                 worth when nothing else in the block can stand in for it.
  leave-one-out  the full representation minus ONE. Says what is lost that no
                 other feature can replace.

Reported together because they answer different questions and disagree in an
informative way. A feature with a large add-one and a small leave-one-out is real
but redundant: something else in the block covers it. Large on both is load
bearing. Small on both is dead weight, and dropping it makes the representation
cheaper and the claim narrower.

The ridge weights themselves are deliberately not the headline. The geometry
features are correlated by construction (five quantiles of one cosine
distribution, four of one norm distribution), and a linear weight split across
correlated columns says more about the correlation than about the feature.
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


def score_block(z, cols, lambdas, n_train):
    """Fit on train, pick the penalty on in-domain validation, report transfer."""
    n = min(n_train, len(z["y_train"]))
    xtr = np.asarray(z["x_train"][:n, cols], dtype=np.float32)
    ytr = z["y_train"][:n]
    mu, sd = standardize(xtr)
    f = lambda a: (np.asarray(a, dtype=np.float64) - mu) / sd      # noqa: E731
    subs = sorted({k[5:] for k in z.files if k.startswith("pb_x_")})
    ws = ridge_path(f(xtr), ytr, lambdas)
    xv, yv = f(z["x_val"][:, cols]), z["y_val"]
    best, best_v = None, -1.0
    for lam in lambdas:
        v = auroc(yv, xv @ ws[lam])
        if v > best_v:
            best_v, best = v, lam
    w = ws[best]
    pb = float(np.mean([auroc(z[f"pb_y_{s}"], f(z[f"pb_x_{s}"][:, cols]) @ w)
                        for s in subs]))
    return pb, float(best_v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, type=Path,
                   help="The winner: content columns then the 20 geometry columns.")
    p.add_argument("--n_train", type=int, default=50000)
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[1e2, 1e3, 1e4, 1e5, 1e6])
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    z = np.load(args.npz)
    total = z["x_train"].shape[1]
    d = total - N_GEOM
    if d <= 0:
        raise SystemExit(f"{args.npz} is {total} wide, too narrow to hold "
                         f"{N_GEOM} geometry features plus content")
    content = list(range(d))
    geom = list(range(d, total))

    base_pb, base_v = score_block(z, content, args.lambdas, args.n_train)
    full_pb, full_v = score_block(z, content + geom, args.lambdas, args.n_train)
    print(f"content only            ProcessBench {base_pb:.4f}  in-domain {base_v:.4f}")
    print(f"content + all geometry  ProcessBench {full_pb:.4f}  in-domain {full_v:.4f}")
    print(f"the whole block is worth {full_pb - base_pb:+.4f}\n")

    rows = []
    print(f"{'feature':<24}{'add-one-in':>12}{'leave-one-out':>15}")
    for j, name in enumerate(GEOM_NAMES):
        add, _ = score_block(z, content + [d + j], args.lambdas, args.n_train)
        loo, _ = score_block(z, content + [c for c in geom if c != d + j],
                             args.lambdas, args.n_train)
        rows.append({"feature": name, "add_one_in": add - base_pb,
                     "leave_one_out": full_pb - loo})
        print(f"{name:<24}{add - base_pb:>+12.4f}{full_pb - loo:>+15.4f}", flush=True)

    print("\nlargest add-one-in (worth something on its own):")
    for r in sorted(rows, key=lambda r: -r["add_one_in"])[:5]:
        print(f"  {r['feature']:<24}{r['add_one_in']:+.4f}")
    print("largest leave-one-out (nothing else replaces it):")
    for r in sorted(rows, key=lambda r: -r["leave_one_out"])[:5]:
        print(f"  {r['feature']:<24}{r['leave_one_out']:+.4f}")
    dead = [r["feature"] for r in rows
            if abs(r["add_one_in"]) < 0.002 and abs(r["leave_one_out"]) < 0.002]
    print(f"dead weight (both under 0.002): {', '.join(dead) if dead else 'none'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"content_only": base_pb, "full": full_pb, "features": rows}, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
