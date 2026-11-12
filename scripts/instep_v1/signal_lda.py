#!/usr/bin/env python3
"""InStep paper, "where the signal lives": a label-supervised 2D view of each verifier's step
representation, fitted on the training steps and drawn on held-out test steps.

Axes (Foley-Sammon discriminant plane, fitted on train40k, features standardized on train):
  x  Fisher's discriminant w1 = Sw^-1 d (d: incorrect minus correct class mean, Sw: pooled
     within-class covariance, shrunk towards a scaled identity by alpha, picked by val AP)
  y  the most discriminant direction orthogonal to w1: w2 = Sw^-1 (d - c w1), w1 . w2 = 0
Drawn on the same class-balanced test steps for every representation (signal_pca.balanced),
each axis in units of the pooled within-class std of those test steps and centred on the correct
steps, so the distance between the two class means reads as a held-out d' on every panel.

Outputs <out>/lda_points_<key>.csv (set, y, x, y2) and <out>/signal_lda.json (alpha, val AP,
held-out AUROC and d' per axis, the class-mean distance in the plane).

    python scripts/instep_v1/signal_lda.py --states $ST8 --extra $SS/prm $SS/reprobe --data $D --out $OUT/signal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.instep_v1.signal_geometry import TRAIN, Data  # noqa: E402
from scripts.instep_v1.signal_pca import KEYS, balanced  # noqa: E402
from scripts.reprobe_cmp.analyze import TEST, auroc, fast_ap  # noqa: E402

ALPHAS = (1e-3, 1e-2, 1e-1, 0.5)


def scatter(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pooled within-class covariance, incorrect minus correct mean) of X."""
    m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for c, m in ((0, m0), (1, m1)):
        Xc = (X[y == c] - m).astype(np.float32)
        Sw += (Xc.T @ Xc).astype(np.float64)
    return Sw / len(X), (m1 - m0).astype(np.float64)


def plane_from(Sw: np.ndarray, d: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    k = Sw.shape[0]
    S = (1 - alpha) * Sw + alpha * np.trace(Sw) / k * np.eye(k)
    w1 = np.linalg.solve(S, d)
    w1 /= np.linalg.norm(w1)
    Si_w1 = np.linalg.solve(S, w1)
    c = (w1 @ np.linalg.solve(S, d)) / (w1 @ Si_w1)
    w2 = np.linalg.solve(S, d) - c * Si_w1
    w2 /= np.linalg.norm(w2)
    return w1, w2


def within_std(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt((z[y == 0].var() + z[y == 1].var()) / 2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--extra", type=Path, nargs="*", default=[])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--val_label", default="ds")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keys", default=KEYS)
    p.add_argument("--per_class", type=int, default=300)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    D = Data(a.states, a.extra, a.data, a.val_label)
    pick = balanced(D, a.per_class)
    sets = np.concatenate([[s] * len(pick[s]) for s in TEST])
    yt = np.concatenate([D.labels(s)[pick[s]] for s in TEST])
    res = []
    for key in a.keys.split(","):
        Xtr, ytr = D.x(TRAIN, key), D.labels(TRAIN)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd
        Sw, d = scatter(Xtr, ytr)
        del Xtr
        Xv, yv = (D.x("val", key) - mu) / sd, D.labels("val")
        fits = {al: plane_from(Sw, d, al) for al in ALPHAS}
        vap = {al: fast_ap(yv, Xv @ w1) for al, (w1, _) in fits.items()}
        al = max(vap, key=vap.get)
        w1, w2 = fits[al]
        X = np.concatenate([(D.x(s, key)[pick[s]] - mu) / sd for s in TEST])
        z1, z2 = X @ w1, X @ w2
        z1 = (z1 - z1[yt == 0].mean()) / within_std(z1, yt)
        z2 = (z2 - z2[yt == 0].mean()) / within_std(z2, yt)
        if z2[yt == 1].mean() < 0:
            z2 = -z2
        dx, dy = z1[yt == 1].mean(), z2[yt == 1].mean()
        r = {"key": key, "alpha": al, "val_ap": vap, "x_auroc": auroc(yt, z1), "y_auroc": auroc(yt, z2),
             "x_dprime": float(dx), "y_dprime": float(dy), "plane_distance": float(np.hypot(dx, dy)),
             "x_auroc_per_set": {s: auroc(yt[sets == s], z1[sets == s]) for s in TEST}}
        res.append(r)
        with open(a.out / f"lda_points_{key}.csv", "w") as f:
            f.write("set,y,x,y2\n")
            for s, c, u, v in zip(sets, yt, z1, z2):
                f.write(f"{s},{int(c)},{u:.4f},{v:.4f}\n")
        print(f"[lda] {key:14s} alpha {al:g} val AP {vap[al]:.3f} | x AUROC {r['x_auroc']:.3f} d' {dx:.2f} | "
              f"y AUROC {r['y_auroc']:.3f} d' {dy:.2f}", flush=True)
    (a.out / "signal_lda.json").write_text(json.dumps({"n": int(len(yt)), "keys": res}, indent=1))


if __name__ == "__main__":
    main()
