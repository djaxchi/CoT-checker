#!/usr/bin/env python3
"""Class separation of the read token and its comparison points (read_token_v1 Stage K).

The figure of scripts/analysis/plot_rep_separation.py, one row per candidate, from
encode directories in the layout of scripts/encode_pooled_multilayer.py (read by
backbone_geometry.load_split): a `train` split (a fixed PRM800K train subsample, the
same items for every candidate), `val` and `test`. Per candidate, fit on train only:

    PCA       top two principal components of the train vectors (no labels)
    LDA       class means and the shrunk pooled within-class covariance, shrinkage
              picked on val AUROC
    LDA x PC  the LDA axis against the top principal component of the train vectors
              with the LDA direction projected out

and drawn on test_2k (1,000 correct, 1,000 incorrect). Writes the figure and a JSON
of the numbers in its titles (test AUROC of LDA and of the best of PC1 / PC2).

    python scripts/analysis/plot_candidate_views.py --out <dir> \
        --cand "Online read token=<dir>:sep:-1" "Strong PRM=<dir>:sep:28" ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.backbone_geometry import load_split  # noqa: E402
from scripts.analysis.plot_prm_geometry import INK, SURF, plt, style  # noqa: E402

SHRINK = (1e-4, 1e-3, 1e-2, 0.1, 0.5)
CLASS = {0: ("#2a78d6", "correct step"), 1: ("#eb6834", "incorrect step")}


def vectors(d: Path, split: str, rep: str, layer: int):
    S, _ = load_split(d, split)
    li = [int(x) for x in S["layers"]].index(layer)
    return S[rep][:, li].astype(np.float64), S["y"].astype(int)


def fit(xtr, ytr, xva, yva):
    """(mu, PCA top 2, LDA direction w, top PC orthogonal to w, shrinkage, val AUROC)."""
    mu = xtr.mean(0)
    xc = xtr - mu
    ev, V = np.linalg.eigh(np.cov(xc, rowvar=False))
    pcs = V[:, ::-1][:, :2]
    m1, m0 = xtr[ytr == 1].mean(0), xtr[ytr == 0].mean(0)
    within = np.concatenate([xtr[ytr == 1] - m1, xtr[ytr == 0] - m0])
    wv, W = np.linalg.eigh(np.cov(within, rowvar=False))
    wv = np.clip(wv, 0, None)
    tr = float(wv.mean())
    de = (m1 - m0) @ W

    def direction(lam):
        return W @ (de / ((1 - lam) * wv + lam * tr))
    va = {lam: roc_auc_score(yva, (xva - mu) @ direction(lam)) for lam in SHRINK}
    lam = max(va, key=va.get)
    w = direction(lam)
    u = w / np.linalg.norm(w)
    r = xc - np.outer(xc @ u, u)
    _, V2 = np.linalg.eigh(np.cov(r, rowvar=False))
    return mu, pcs, w, V2[:, -1], lam, va[lam]


def lims(P, q=0.5):
    lo, hi = np.percentile(P, [q, 100 - q], axis=0)
    pad = 0.05 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cand", nargs="+", required=True, help="label=dir:rep:layer")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cands = []
    for spec in a.cand:
        label, rest = spec.split("=", 1)
        d, rep, layer = rest.rsplit(":", 2)
        cands.append((label, Path(d), rep, int(layer)))

    fig, axes = plt.subplots(len(cands), 3, figsize=(13.5, 3.7 * len(cands)), facecolor=SURF, squeeze=False)
    res, proj = [], {}
    for r, (label, d, rep, layer) in enumerate(cands):
        xtr, ytr = vectors(d, "train", rep, layer)
        xva, yva = vectors(d, "val", rep, layer)
        xte, y = vectors(d, "test", rep, layer)
        mu, pcs, w, v, lam, va = fit(xtr, ytr, xva, yva)
        pca = (xte - mu) @ pcs
        score = (xte - mu) @ w
        lda2 = np.stack([score, (xte - mu) @ v], 1)
        au = float(roc_auc_score(y, score))
        pc_au = [float(max(t, 1 - t)) for t in (roc_auc_score(y, pca[:, 0]), roc_auc_score(y, pca[:, 1]))]
        res.append({"label": label, "dim": int(xtr.shape[1]), "n_train": int(len(ytr)), "shrink": lam,
                    "val_auroc": float(va), "lda_test_auroc": au, "pc1_auroc": pc_au[0], "pc2_auroc": pc_au[1]})
        print(f"[views] {label}: train {len(ytr)}, dim {xtr.shape[1]}, LDA test AUROC {au:.4f} "
              f"(shrink {lam}), PC1 {pc_au[0]:.3f}, PC2 {pc_au[1]:.3f}", flush=True)
        proj[f"pca_{r}"], proj[f"lda2_{r}"], proj[f"y_{r}"] = pca, lda2, y
        order = np.random.default_rng(a.seed).permutation(len(y))
        order = np.concatenate([order[y[order] == 0], order[y[order] == 1]])
        for c, (P, title, xl, yl) in enumerate([
                (pca, f"top-2 PCs (unsupervised), best PC AUROC {max(pc_au):.3f}", "PC1", "PC2"),
                (lda2, "LDA axis x top orthogonal PC", "LDA axis (fit on PRM800K train)", "PC1 orthogonal to LDA")]):
            ax = axes[r, c]
            style(ax)
            ax.grid(False)
            ax.scatter(P[order, 0], P[order, 1], s=6, c=[CLASS[k][0] for k in y[order]], alpha=0.55, linewidths=0)
            xlim, ylim = lims(P)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(xl, color=INK, fontsize=8.5)
            ax.set_ylabel(yl, color=INK, fontsize=8.5)
            ax.set_title(f"{label}\n{title}", color=INK, fontsize=9)
        ax = axes[r, 2]
        style(ax)
        bins = np.linspace(np.percentile(score, 0.5), np.percentile(score, 99.5), 60)
        for k in (0, 1):
            ax.hist(score[y == k], bins=bins, color=CLASS[k][0], alpha=0.55, density=True,
                    label=f"{CLASS[k][1]} (n={int((y == k).sum())})")
        ax.set_title(f"{label}\nLDA score on PRM800K test, AUROC {au:.3f}", color=INK, fontsize=9)
        ax.set_xlabel("LDA projection", color=INK, fontsize=8.5)
        ax.set_ylabel("density", color=INK, fontsize=8.5)
        if r == 0:
            ax.legend(frameon=False, fontsize=7.5)
    np.savez(a.out / "candidate_views_proj.npz", labels=np.array([c[0] for c in cands]), **proj)
    fig.tight_layout()
    fig.savefig(a.out / "candidate_views.png", dpi=140, facecolor=SURF)
    plt.close(fig)
    (a.out / "candidate_views.json").write_text(json.dumps(res, indent=1))
    print(f"[views] wrote {a.out / 'candidate_views.png'}", flush=True)


if __name__ == "__main__":
    main()
