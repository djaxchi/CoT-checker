"""Hunt for a representation that parts correct from incorrect, beyond the one-layer linear
probe. Two cheap ideas, both on the already-extracted PRM800K heldout features (no model):

  (A) class-conditional Gaussian density (QDA / Mahalanobis):
      fit a Gaussian to each class in PCA space and score by which is more likely. Unlike a
      linear probe this uses the covariance shape, so it can catch errors that are
      off-manifold rather than across a single hyperplane.

  (B) cross-layer probe trajectory:
      run a linear probe at every stored layer, stack the out-of-fold scores into a per-step
      trajectory, and (1) plot the mean trajectory per class to see separation grow with
      depth, (2) test whether using all layers at once beats the best single layer.

All scores are 5-fold out-of-fold (cross_val_predict), so the histograms are honest and not
train-set memorisation. AUC reported with the existing linear probe as the baseline to beat.

Outputs (results/repr_separation/):
  - mahalanobis_L{layer}.png   class-conditional density separation + method AUC bars
  - crosslayer_probe.png       mean probe-score trajectory per class + per-layer/joint AUC
  - repr_separation.json       every number

Usage:
    python scripts/analysis/s3_repr_separation.py
    python scripts/analysis/s3_repr_separation.py --maha_layer 28
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.prm800k_val_data import load_prm800k_multitoken

PRM_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b/prm_multitoken")
STEM = "prm800k_heldout_test"
ROOT = Path("results/repr_separation")
SEED = 42
GREEN, RED = "#3cb44b", "#e6194B"


def oof_scores(estimator, X, y, cv, use_proba):
    """Out-of-fold score for the positive class (1 = incorrect)."""
    if use_proba:
        p = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
        return p[:, 1]
    return cross_val_predict(estimator, X, y, cv=cv, method="decision_function")


def load_layer(layer):
    return load_prm800k_multitoken(PRM_DIR, STEM, layer, "last").hidden.astype(np.float32)


def run_mahalanobis(layer, y, cv, n_pca, out_dir):
    X = load_layer(layer)
    methods = {
        "linear probe (LogReg)": (make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            False),
        "LDA (shared cov)": (make_pipeline(
            StandardScaler(), PCA(n_pca, random_state=SEED), LinearDiscriminantAnalysis()),
            True),
        "QDA / Mahalanobis density": (make_pipeline(
            StandardScaler(), PCA(n_pca, random_state=SEED),
            QuadraticDiscriminantAnalysis(reg_param=0.4)), True),
        "kNN density (k=50)": (make_pipeline(
            StandardScaler(), PCA(n_pca, random_state=SEED),
            KNeighborsClassifier(n_neighbors=50)), True),
    }
    scores, aucs = {}, {}
    for name, (est, use_proba) in methods.items():
        s = oof_scores(est, X, y, cv, use_proba)
        scores[name] = s
        aucs[name] = float(roc_auc_score(y, s))
        print(f"  [maha L{layer}] {name:28s} AUC={aucs[name]:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, name in zip(ax[:2], ["QDA / Mahalanobis density", "linear probe (LogReg)"]):
        s = scores[name]
        a.hist(s[y == 0], bins=50, alpha=0.6, color=GREEN, label="correct")
        a.hist(s[y == 1], bins=50, alpha=0.6, color=RED, label="incorrect")
        a.set_title(f"{name}\nout-of-fold score  AUC={aucs[name]:.3f}")
        a.set_xlabel("score (high = incorrect)"); a.legend()
    names = list(aucs)
    ax[2].barh(range(len(names)), [aucs[n] for n in names],
               color=["#4363d8", "#42d4f4", "#911eb4", "#f58231"])
    ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels(names, fontsize=9)
    ax[2].axvline(0.5, color="#888", ls=":")
    ax[2].set_xlim(0.5, max(aucs.values()) + 0.03); ax[2].set_xlabel("AUROC")
    for i, n in enumerate(names):
        ax[2].text(aucs[n] + 0.002, i, f"{aucs[n]:.3f}", va="center", fontsize=9)
    ax[2].set_title(f"L{layer}: does non-linear geometry beat the linear probe?")
    fig.suptitle(f"Class-conditional density separation - PRM800K {STEM}  L{layer}/last "
                 f"(n={len(y)}, 1=incorrect)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = out_dir / f"mahalanobis_L{layer}.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    print(f"  [maha] -> {png}")
    return aucs


def run_crosslayer(layers, y, cv, out_dir):
    S = np.zeros((len(y), len(layers)), dtype=np.float32)
    per_layer_auc = {}
    for j, L in enumerate(layers):
        X = load_layer(L)
        est = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        S[:, j] = oof_scores(est, X, y, cv, use_proba=False)
        per_layer_auc[L] = float(roc_auc_score(y, S[:, j]))
        print(f"  [xlayer] L{L:>2} single-probe AUC={per_layer_auc[L]:.4f}")

    # joint: meta-probe on the 6 out-of-fold scores (mild optimism; flagged)
    meta = oof_scores(LogisticRegression(max_iter=2000, class_weight="balanced"),
                      S, y, cv, use_proba=False)
    meta_auc = float(roc_auc_score(y, meta))
    best_single = max(per_layer_auc.values())
    print(f"  [xlayer] best single AUC={best_single:.4f}  joint(6-layer) AUC={meta_auc:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.array(layers)
    for cls, c, lab in [(0, GREEN, "correct"), (1, RED, "incorrect")]:
        m = y == cls
        mu = S[m].mean(0); ci = 1.96 * S[m].std(0, ddof=1) / np.sqrt(m.sum())
        ax[0].plot(x, mu, "-o", color=c, label=lab)
        ax[0].fill_between(x, mu - ci, mu + ci, color=c, alpha=0.2)
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("out-of-fold probe score")
    ax[0].set_title("mean probe-score trajectory per class\n(separation grows with depth?)")
    ax[0].legend(); ax[0].set_xticks(x)

    ax[1].hist(meta[y == 0], bins=50, alpha=0.6, color=GREEN, label="correct")
    ax[1].hist(meta[y == 1], bins=50, alpha=0.6, color=RED, label="incorrect")
    ax[1].set_title(f"joint 6-layer meta-probe\nAUC={meta_auc:.3f}")
    ax[1].set_xlabel("score (high = incorrect)"); ax[1].legend()

    labels = [f"L{L}" for L in layers] + ["joint\n6-layer"]
    vals = [per_layer_auc[L] for L in layers] + [meta_auc]
    ax[2].bar(range(len(vals)), vals,
              color=["#4363d8"] * len(layers) + ["#911eb4"])
    ax[2].set_xticks(range(len(vals))); ax[2].set_xticklabels(labels, fontsize=8)
    ax[2].axhline(best_single, color="#888", ls=":", label=f"best single {best_single:.3f}")
    ax[2].set_ylim(0.5, max(vals) + 0.02); ax[2].set_ylabel("AUROC"); ax[2].legend(fontsize=8)
    ax[2].set_title("per-layer vs joint")
    fig.suptitle(f"Cross-layer probe trajectory - PRM800K {STEM} (n={len(y)})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = out_dir / "crosslayer_probe.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    print(f"  [xlayer] -> {png}")
    return {"per_layer_auc": per_layer_auc, "best_single": best_single, "joint_auc": meta_auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maha_layer", type=int, default=20)
    ap.add_argument("--n_pca", type=int, default=256)
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((PRM_DIR / f"{STEM}_manifest.json").read_text())
    layers = list(manifest["layer_indices"])
    y = load_prm800k_multitoken(PRM_DIR, STEM, layers[0], "last").label.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"[repr-sep] n={len(y)} incorrect={int(y.sum())} layers={layers} pca={args.n_pca}")

    print("\n(A) class-conditional density:")
    maha = run_mahalanobis(args.maha_layer, y, cv, args.n_pca, args.out_dir)
    print("\n(B) cross-layer probe trajectory:")
    xlayer = run_crosslayer(layers, y, cv, args.out_dir)

    (args.out_dir / "repr_separation.json").write_text(json.dumps(
        {"n": int(len(y)), "incorrect": int(y.sum()), "layers": layers,
         "maha_layer": args.maha_layer, "n_pca": args.n_pca,
         "mahalanobis_auc": maha, "crosslayer": xlayer}, indent=2))
    print(f"\n[done] -> {args.out_dir/'repr_separation.json'}")


if __name__ == "__main__":
    main()
