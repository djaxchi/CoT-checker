"""Resolve the apparent paradox: linear probe separates (~0.74) but UMAP shows nothing.

There is no contradiction. UMAP is unsupervised and lays points out by their dominant
(high-variance) directions, which here are topic/problem/length and are orthogonal to the
thin correctness direction. The probe is supervised and reads that one low-variance
direction directly. This script shows both views side by side on the same points:

  left   UMAP(cosine) coloured by label              -> no separation (variance view)
  middle supervised projection: x = out-of-fold      -> separation along x (label view)
         probe score, y = top PC orthogonal to it
  right  the 1D probe margin with d-prime + AUC

It also prints the crux number: the fraction of total variance that lies along the probe
direction (tiny), which is exactly why a variance-preserving 2D map never surfaces it.

Output: results/prm800k_layers/supervised_view_L{layer}_{token}.png

Usage:
    python scripts/analysis/s3_prm800k_supervised_view.py            # L20 / last
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, normalize
from umap import UMAP

from src.data.prm800k_val_data import load_prm800k_multitoken

DEFAULT_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b/prm_multitoken")
ROOT = Path("results/prm800k_layers")
LABEL_COLOR = {0: "#3cb44b", 1: "#e6194B"}        # correct / incorrect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--stem", type=str, default="prm800k_heldout_test")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--token", type=str, default="last")
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = load_prm800k_multitoken(args.merged_dir, args.stem, args.layer, args.token)
    y = d.label.astype(int)
    Xs = StandardScaler().fit_transform(d.hidden).astype(np.float32)
    P = Xs.shape[1]

    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
    # out-of-fold score = honest supervised x-axis (no leakage)
    score = cross_val_predict(clf, Xs, y, cv=5, method="decision_function")
    auc = roc_auc_score(y, score)
    m0, m1 = score[y == 0].mean(), score[y == 1].mean()
    sd = np.sqrt(0.5 * (score[y == 0].var() + score[y == 1].var()))
    dprime = (m1 - m0) / sd

    # probe axis (full fit, just to build an orthogonal basis), unit length
    w = clf.fit(Xs, y).coef_.ravel()
    w_unit = w / np.linalg.norm(w)
    proj = Xs @ w_unit
    var_probe = float(proj.var())
    var_total = float(Xs.var(axis=0).sum())          # ~P for standardized dims
    frac = var_probe / var_total

    # y-axis: dominant variance AFTER removing the probe direction
    Xperp = Xs - np.outer(proj, w_unit)
    pc1 = PCA(n_components=1, random_state=42).fit_transform(Xperp).ravel()

    # unsupervised reference: cosine UMAP on the same points
    emb = UMAP(n_components=2, random_state=42, metric="cosine",
               n_neighbors=20, min_dist=0.0).fit_transform(normalize(Xs))

    print(f"[plane] L{args.layer}/{args.token}  AUC={auc:.3f}  d'={dprime:.2f} SD apart")
    print(f"[variance] probe direction holds {var_probe:.1f} of {var_total:.0f} total "
          f"variance = {100*frac:.3f}%  (top PC of residual holds {pc1.var():.1f})")
    print(f"[reading] one supervised direction with {100*frac:.2f}% of the variance "
          f"carries the whole signal; UMAP spends its 2 dims on the other 99.9%.")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    for c in (0, 1):
        mk = y == c
        lab = "correct" if c == 0 else "incorrect"
        ax[0].scatter(emb[mk, 0], emb[mk, 1], s=5, alpha=0.5, color=LABEL_COLOR[c], label=lab)
        ax[1].scatter(score[mk], pc1[mk], s=5, alpha=0.5, color=LABEL_COLOR[c], label=lab)
    ax[0].set_title("UNSUPERVISED  UMAP(cosine)\nno separation: variance is topic, not label")
    ax[0].set_xlabel("UMAP-1"); ax[0].set_ylabel("UMAP-2"); ax[0].legend()
    ax[1].set_title(f"SUPERVISED  probe axis x top residual-PC\n"
                    f"separation along x  (AUC {auc:.2f}, d'={dprime:.2f})")
    ax[1].set_xlabel("probe score (out-of-fold)")
    ax[1].set_ylabel("top variance dir orthogonal to probe"); ax[1].legend()

    ax[2].hist(score[y == 0], bins=50, alpha=0.6, color=LABEL_COLOR[0], label="correct")
    ax[2].hist(score[y == 1], bins=50, alpha=0.6, color=LABEL_COLOR[1], label="incorrect")
    ax[2].axvline(m0, color=LABEL_COLOR[0], ls="--"); ax[2].axvline(m1, color=LABEL_COLOR[1], ls="--")
    ax[2].set_title(f"the 1D margin: means {dprime:.2f} SD apart\n"
                    f"probe direction = {100*frac:.2f}% of total variance")
    ax[2].set_xlabel("probe score"); ax[2].legend()

    fig.suptitle(f"Why a linear probe separates but UMAP cannot - PRM800K "
                 f"L{args.layer}/{args.token} (n={len(y)}, {P} dims)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.out_dir / f"supervised_view_L{args.layer}_{args.token}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
