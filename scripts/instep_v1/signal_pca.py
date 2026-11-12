#!/usr/bin/env python3
"""InStep paper, "where the signal lives": an unsupervised 2D view of each verifier's step
representation on comparable data, so that correctness can be the largest variance.

Comparable data: the same test steps for every representation (signal_geometry.Data joint
mask), class-balanced inside each of the 8 test sets (--per_class incorrect and as many correct
steps per set, sampled once with seed 0), and each set centred on its own mean so the variance
between tasks is removed. Each representation is then standardized per dimension on that sample
(--scale dim; none keeps the raw scale) and reduced by PCA (no labels used in the PCA). PCs are signed so incorrect steps score higher.

Outputs <out>/pca_<scale>_points_<key>.csv (set, y, pc1, pc2) and <out>/signal_pca_<scale>.json with, per key,
the variance share and AUROC of the top --n_pcs PCs, and the class-mean separation along PC1
(d' = mean difference / pooled std).

    python scripts/instep_v1/signal_pca.py --states $ST8 --extra $SS/prm $SS/reprobe --data $D --out $OUT/signal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.instep_v1.signal_geometry import Data  # noqa: E402
from scripts.reprobe_cmp.analyze import TEST, auroc  # noqa: E402

KEYS = "passive_final,reprobe_rep,prm_final,sep"


def balanced(D: Data, per_class: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Row indices (into the masked test rows of each set), the same for every key."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in TEST:
        y = D.labels(s)
        n = min(per_class, int((y == 1).sum()), int((y == 0).sum()))
        idx = [rng.choice(np.flatnonzero(y == c), n, replace=False) for c in (0, 1)]
        out[s] = np.sort(np.concatenate(idx))
    return out


def project(X: np.ndarray, sets: np.ndarray, y: np.ndarray, n_pcs: int, scale: str = "dim") -> tuple[np.ndarray, np.ndarray]:
    """Per-set centring, scaling (dim: each dimension to unit variance; none: raw), PCA; returns
    (scores signed so incorrect > correct, variance share of each PC)."""
    X = X.astype(np.float64)
    for s in np.unique(sets):
        X[sets == s] -= X[sets == s].mean(0)
    if scale == "dim":
        X /= X.std(0) + 1e-6
    U, S, _ = np.linalg.svd(X.astype(np.float32), full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    Z = U[:, :n_pcs] * S[:n_pcs]
    sign = np.sign(Z[y == 1].mean(0) - Z[y == 0].mean(0))
    return Z * np.where(sign == 0, 1, sign), var[:n_pcs]


def dprime(z: np.ndarray, y: np.ndarray) -> float:
    a, b = z[y == 0], z[y == 1]
    return float((b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--extra", type=Path, nargs="*", default=[])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--val_label", default="ds")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keys", default=KEYS)
    p.add_argument("--per_class", type=int, default=300)
    p.add_argument("--n_pcs", type=int, default=10)
    p.add_argument("--scale", choices=["dim", "none"], default="dim")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    D = Data(a.states, a.extra, a.data, a.val_label)
    pick = balanced(D, a.per_class)
    sets = np.concatenate([[s] * len(pick[s]) for s in TEST])
    y = np.concatenate([D.labels(s)[pick[s]] for s in TEST])
    print(f"[pca] {len(y)} steps, per set: " + ", ".join(f"{s} {len(pick[s])}" for s in TEST), flush=True)
    res = []
    for key in a.keys.split(","):
        X = np.concatenate([D.x(s, key)[pick[s]] for s in TEST])
        Z, var = project(X, sets, y, a.n_pcs, a.scale)
        aur = [auroc(y, Z[:, j]) for j in range(Z.shape[1])]
        r = {"key": key, "dim": int(X.shape[1]), "var_share": var.tolist(), "pc_auroc": aur,
             "pc1_dprime": dprime(Z[:, 0], y), "pc1_auroc_per_set": {s: auroc(y[sets == s], Z[sets == s, 0]) for s in TEST}}
        res.append(r)
        with open(a.out / f"pca_{a.scale}_points_{key}.csv", "w") as f:
            f.write("set,y,pc1,pc2\n")
            for s, c, z1, z2 in zip(sets, y, Z[:, 0], Z[:, 1]):
                f.write(f"{s},{int(c)},{z1:.4f},{z2:.4f}\n")
        print(f"[pca] {key:14s} dim {r['dim']:5d} | PC1 var {var[0]:.3f} AUROC {aur[0]:.3f} d' {r['pc1_dprime']:.2f} | "
              f"PC2 var {var[1]:.3f} AUROC {aur[1]:.3f} | best PC {int(np.argmax(aur)) + 1} ({max(aur):.3f})", flush=True)
    (a.out / f"signal_pca_{a.scale}.json").write_text(json.dumps({"scale": a.scale, "n": int(len(y)), "per_set": {s: int(len(pick[s])) for s in TEST},
                                                       "keys": res}, indent=1))


if __name__ == "__main__":
    main()
