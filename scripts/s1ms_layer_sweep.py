#!/usr/bin/env python3
"""Phase 1: where does the correctness signal live? Per-layer probe + subset.

For each layer in a multi-layer cache (encode_prm800k_multilayer.py), train the
same DenseLinear-style probe (standardized logistic) on probe_train_40k and
evaluate macro step F1 on val_1k, then sweep the top-k most important units to
find the minimal sufficient subset at that layer. Produces:

  layer_f1_vs_depth.png      F1 vs layer fraction (settles "do mid layers carry
                             correctness as well as the last layer?")
  layer_subset_f1_vs_k.png   F1 vs #units, one line per layer
  layer_sweep.json           per-layer F1, chosen k*, and the unit indices
                             (these feed the steering experiment)

CPU only. Reuses merged multi-layer cache; needs sklearn + matplotlib.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DEFAULT_KS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def _log(m): print(f"[sweep {time.strftime('%H:%M:%S')}] {m}", flush=True)


def fit_eval(xtr, ytr, xte, yte):
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(xtr), ytr)
    f1 = float(f1_score(yte, clf.predict(sc.transform(xte)), average="macro"))
    return f1, clf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", type=Path, required=True, help="Multi-layer cache dir.")
    p.add_argument("--train_stem", default="probe_train_40k")
    p.add_argument("--val_stem", default="val_1k")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--tol", type=float, default=0.01, help="k* = smallest k within tol of full-layer F1.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cache_dir / "multilayer_manifest.json").read_text())
    layer_idx_by_frac = manifest["layer_indices"]  # {"0.10": idx, ...}
    fracs = sorted(layer_idx_by_frac.keys(), key=float)

    ytr = np.load(args.cache_dir / f"{args.train_stem}_y.npy").astype(int)
    yte = np.load(args.cache_dir / f"{args.val_stem}_y.npy").astype(int)

    per_layer = {}
    for fr in fracs:
        li = layer_idx_by_frac[fr]
        xtr = np.load(args.cache_dir / f"{args.train_stem}_L{li}_h.npy").astype(np.float32)
        xte = np.load(args.cache_dir / f"{args.val_stem}_L{li}_h.npy").astype(np.float32)
        full_f1, clf = fit_eval(xtr, ytr, xte, yte)
        rank = np.argsort(-np.abs(clf.coef_.reshape(-1)))
        ks, f1s = [], []
        for k in args.ks:
            k = min(k, xtr.shape[1]); cols = rank[:k]
            f1k, _ = fit_eval(xtr[:, cols], ytr, xte[:, cols], yte)
            ks.append(k); f1s.append(f1k)
        kstar = next((k for k, f in zip(ks, f1s) if f >= full_f1 - args.tol), ks[-1])
        cols_star = [int(i) for i in rank[:kstar]]
        per_layer[fr] = {"layer_index": li, "full_f1": full_f1, "ks": ks, "f1s": f1s,
                         "kstar": kstar, "unit_indices_kstar": cols_star}
        _log(f"frac={fr} L{li}: full F1={full_f1:.4f}  k*={kstar} (subset reaches within {args.tol})")

    (args.out_dir / "layer_sweep.json").write_text(json.dumps(
        {"model_name": manifest.get("model_name"), "per_layer": per_layer}, indent=2))

    # F1 vs depth
    xs = [float(fr) for fr in fracs]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(xs, [per_layer[fr]["full_f1"] for fr in fracs], "o-", color="#225522")
    ax.set_xlabel("layer depth (fraction)"); ax.set_ylabel("val macro step F1")
    ax.set_title(f"Correctness F1 vs layer depth  ({manifest.get('model_name','')})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "layer_f1_vs_depth.png", dpi=150); plt.close(fig)
    _log(f"wrote {args.out_dir/'layer_f1_vs_depth.png'}")

    # subset F1 vs k, one line per layer
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for i, fr in enumerate(fracs):
        d = per_layer[fr]
        ax.plot(d["ks"], d["f1s"], "o-", color=cmap(i / max(len(fracs) - 1, 1)), label=f"{fr}")
    ax.set_xscale("log"); ax.set_xlabel("# units (top by |weight|)"); ax.set_ylabel("val macro step F1")
    ax.set_title("Minimal subset per layer"); ax.legend(fontsize=7, title="depth"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "layer_subset_f1_vs_k.png", dpi=150); plt.close(fig)
    _log(f"wrote {args.out_dir/'layer_subset_f1_vs_k.png'}")
    _log("done")


if __name__ == "__main__":
    main()
