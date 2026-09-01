"""Rank representations with a probe that has no arbitrary budget in it.

The convergence sweep found that the screen's verdict was mostly about its
budget. Training longer made ProcessBench transfer monotonically WORSE for every
representation (`dir` 0.7603 at 8 epochs, 0.7267 at 60), and the ranking at the
incumbent budget anti-correlated with the ranking at every other budget, Spearman
-0.07 to -0.24. The stacked representation was best at 8 epochs and worst at 60.
So "which representation is better" was being decided by where an unregularised
SGD run happened to be stopped.

The fix is not a better stopping rule, it is removing the stopping rule. Ridge
regression on +/-1 labels has a closed form, no randomness, no epochs, and one
honest knob:

    w = (X'X + lambda I)^-1 X'y

That knob is the interesting one rather than a nuisance. As lambda goes to zero
this approaches LDA, which whitens by the within-class covariance. As lambda
grows it approaches the mean-difference direction, the centroid rule. Those are
exactly the two things the conicity study compared when it found 0.63 for the
centroid rule against 0.82 whitened and concluded the gap was the metric, not the
direction. So sweeping lambda traces the path between them, and each
representation gets to be read at its own best point on that path instead of at
one arbitrary point on someone's optimiser trajectory.

Reported per representation, following the project's convention of a selected
value plus a ceiling:

  val-selected   lambda chosen by in-domain validation AUROC, the honest number,
                 since ProcessBench is the held-out domain and may not be peeked at
  oracle         the best ProcessBench score over the lambda path, a ceiling
  in-domain      at the selected lambda, so source-domain fit and transfer can be
                 read against each other

If in-domain rises while ProcessBench falls along the path, the decay the sweep
found is the probe overfitting the source domain, and the gap between the
val-selected and oracle columns says how much that costs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_representation import auroc, standardize  # noqa: E402
from stratified_auroc import stratified_auroc  # noqa: E402


def ridge_path(x: np.ndarray, y: np.ndarray, lambdas) -> dict[float, np.ndarray]:
    """Solve for every lambda from one eigendecomposition of the Gram matrix.

    Each solve reuses X'X and X'y, so the whole path costs one pass over the data
    plus one symmetric eigendecomposition, not one solve per lambda.
    """
    xd = np.asarray(x, dtype=np.float64)
    t = np.where(np.asarray(y) > 0.5, 1.0, -1.0)
    g = xd.T @ xd
    b = xd.T @ t
    evals, evecs = np.linalg.eigh(g)
    bp = evecs.T @ b
    return {lam: evecs @ (bp / (evals + lam)) for lam in lambdas}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", nargs="+", required=True, type=Path)
    p.add_argument("--n_train", type=int, default=50000)
    p.add_argument("--max_dim", type=int, default=9000,
                   help="The eigendecomposition is cubic in width; wider inputs "
                        "are skipped loudly rather than left to run for hours.")
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    p.add_argument("--n_bins", type=int, default=50)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    rows = []
    print(f"{'representation':<24}{'val-sel PB':>11}{'oracle PB':>10}"
          f"{'in-dom':>8}{'within-len':>11}{'lambda':>9}{'dim':>7}")
    for f in args.npz:
        z = np.load(f)
        n = min(args.n_train, len(z["y_train"]))
        xtr, ytr = z["x_train"][:n], z["y_train"][:n]
        if xtr.shape[1] > args.max_dim:
            print(f"{f.stem:<24} skipped, dim {xtr.shape[1]} over --max_dim")
            continue
        mu, sd = standardize(xtr)
        zs = lambda a: (np.asarray(a, dtype=np.float64) - mu) / sd     # noqa: E731
        subs = sorted({k[5:] for k in z.files if k.startswith("pb_x_")})
        ws = ridge_path(zs(xtr), ytr, args.lambdas)
        xv, yv = zs(z["x_val"]), z["y_val"]
        has_len = f"pb_len_{subs[0]}" in z.files

        path = []
        for lam in args.lambdas:
            w = ws[lam]
            pb = [auroc(z[f"pb_y_{s}"], zs(z[f"pb_x_{s}"]) @ w) for s in subs]
            path.append({"lambda": lam, "in_domain": auroc(yv, xv @ w),
                         "pb": float(np.mean(pb))})
        best = max(path, key=lambda r: r["in_domain"])
        oracle = max(r["pb"] for r in path)
        w = ws[best["lambda"]]
        wl = float(np.mean([
            stratified_auroc(z[f"pb_y_{s}"], zs(z[f"pb_x_{s}"]) @ w,
                             z[f"pb_len_{s}"], args.n_bins) for s in subs
        ])) if has_len else float("nan")
        r = {"name": f.stem, "dim": int(xtr.shape[1]), "path": path,
             "val_selected_pb": best["pb"], "oracle_pb": oracle,
             "in_domain": best["in_domain"], "within_length": wl,
             "lambda": best["lambda"]}
        rows.append(r)
        print(f"{r['name']:<24}{r['val_selected_pb']:>11.4f}{r['oracle_pb']:>10.4f}"
              f"{r['in_domain']:>8.4f}{r['within_length']:>11.4f}"
              f"{r['lambda']:>9.0e}{r['dim']:>7d}", flush=True)

    rows.sort(key=lambda r: -r["val_selected_pb"])
    print(f"\nranked by the val-selected column, which never sees ProcessBench:")
    for r in rows:
        print(f"  {r['name']:<24}{r['val_selected_pb']:>9.4f}")
    if rows:
        w = rows[0]
        print(f"\npath for {w['name']}, in-domain against transfer:")
        for e in w["path"]:
            print(f"  lambda {e['lambda']:>8.0e}   in-domain {e['in_domain']:.4f}"
                  f"   processbench {e['pb']:.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
