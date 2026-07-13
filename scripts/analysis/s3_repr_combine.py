"""Bake-off of strategies for COMBINING representations to separate correct from incorrect.

Builds on s3_repr_separation.py, which found multi-layer stacking lifts AUROC 0.80 -> ~0.85.
Here we try many ways to combine, all on the same PRM800K heldout features (6 layers x 2
tokens already on disk), paired 5-fold out-of-fold so the numbers compare directly.

Families:
  early fusion : concatenate raw features, then one linear model (leakage-free)
      - concat layers (last); concat layers (last+first); + L2 strength sweep
      - concat + PCA + LDA (covariance whitening); concat + PCA + LogReg
      - consecutive-layer deltas (what each block adds), concatenated
  late fusion  : combine per-layer probe outputs
      - mean of per-layer probabilities (leakage-free ensemble)
      - meta LogReg on per-layer scores (stacking; mild optimism, flagged)
      - meta GradientBoosting on scores (non-linear layer interactions; flagged)
      - cross-layer trajectory shape features (mean/std/slope/min/max; flagged)

Leakage-free strategies use a single cross_val_predict. Score-stacking strategies refit on
out-of-fold scores and are marked optimistic (*) since they peek slightly; the early-fusion
concat numbers are the trustworthy headline.

Outputs (results/repr_combine/):
  - combine_ranking.png   ranked AUROC bars (green=leakage-free, orange=optimistic)
  - combine_best.png      separation histogram of the best leakage-free strategy
  - repr_combine.json     every number

Usage: python scripts/analysis/s3_repr_combine.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.prm800k_val_data import load_prm800k_multitoken

PRM_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b/prm_multitoken")
STEM = "prm800k_heldout_test"
ROOT = Path("results/repr_combine")
SEED = 42
GREEN, RED = "#3cb44b", "#e6194B"

_CACHE: dict = {}


def feat(layer, token):
    k = (layer, token)
    if k not in _CACHE:
        _CACHE[k] = load_prm800k_multitoken(PRM_DIR, STEM, layer, token).hidden.astype(np.float32)
    return _CACHE[k]


def logreg(C=1.0):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1500, class_weight="balanced", C=C))


def oof_dec(est, X, y, cv):
    return cross_val_predict(est, X, y, cv=cv, method="decision_function")


def oof_proba(est, X, y, cv):
    return cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PRM_DIR / f"{STEM}_manifest.json").read_text())
    layers = list(manifest["layer_indices"])
    y = load_prm800k_multitoken(PRM_DIR, STEM, layers[0], "last").label.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    n = len(y)
    print(f"[combine] n={n} incorrect={int(y.sum())} layers={layers}")

    results = {}   # name -> (auc, leakage_free, score_array)

    def add(name, score, leak_free):
        a = float(roc_auc_score(y, score))
        results[name] = (a, leak_free, score)
        print(f"  {'   ' if leak_free else ' * '}{name:34s} AUC={a:.4f}", flush=True)

    def add_safe(name, fn, leak_free):
        try:
            add(name, fn(), leak_free)
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}", flush=True)

    # ---- per-layer baselines + reusable OOF outputs (last token) ------------
    Sdec = np.zeros((n, len(layers)), np.float32)   # decision scores
    Pp = np.zeros((n, len(layers)), np.float32)     # probabilities
    for j, L in enumerate(layers):
        Sdec[:, j] = oof_dec(logreg(), feat(L, "last"), y, cv)
        Pp[:, j] = 1 / (1 + np.exp(-Sdec[:, j]))
        add(f"single L{L} (last)", Sdec[:, j], True)
    best_single = max(results[f"single L{L} (last)"][0] for L in layers)

    # ---- early fusion: concat raw features (leakage-free) -------------------
    concat_last = np.concatenate([feat(L, "last") for L in layers], axis=1)
    concat_all = np.concatenate([feat(L, t) for L in layers for t in ("first", "last")], axis=1)
    for C in (0.05, 0.2, 1.0):
        add_safe(f"concat last, LogReg C={C}",
                 lambda C=C: oof_dec(logreg(C), concat_last, y, cv), True)
    add_safe("concat last+first, LogReg",
             lambda: oof_dec(logreg(0.2), concat_all, y, cv), True)
    add_safe("concat last, PCA256+LogReg",
             lambda: oof_dec(make_pipeline(StandardScaler(), PCA(256, random_state=SEED),
                             LogisticRegression(max_iter=1500, class_weight="balanced")),
                             concat_last, y, cv), True)
    # robust whitening: shrinkage LDA (lsqr) instead of the svd solver that fails to converge
    add_safe("concat last, PCA256+LDA (whiten)",
             lambda: oof_proba(make_pipeline(StandardScaler(), PCA(256, random_state=SEED),
                              LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                              concat_last, y, cv), True)

    # consecutive-layer deltas (what each block writes)
    deltas = np.concatenate([feat(layers[i + 1], "last") - feat(layers[i], "last")
                             for i in range(len(layers) - 1)], axis=1)
    add_safe("layer deltas concat, LogReg",
             lambda: oof_dec(logreg(0.2), deltas, y, cv), True)

    # ---- late fusion -------------------------------------------------------
    add_safe("mean per-layer proba (ensemble)", lambda: Pp.mean(axis=1), True)  # leakage-free
    add_safe("stack: meta LogReg on scores",
             lambda: oof_dec(LogisticRegression(max_iter=1500, class_weight="balanced"),
                             Sdec, y, cv), False)
    add_safe("stack: meta GBoost on scores",
             lambda: oof_proba(GradientBoostingClassifier(
                 n_estimators=200, max_depth=3, random_state=SEED), Sdec, y, cv), False)
    shape = np.column_stack([Sdec.mean(1), Sdec.std(1), Sdec.max(1), Sdec.min(1),
                             Sdec[:, -1] - Sdec[:, 0],
                             np.polyfit(np.array(layers), Sdec.T, 1)[0]])  # slope per row
    add_safe("trajectory-shape feats, LogReg", lambda: oof_dec(logreg(), shape, y, cv), False)

    # ---- ranking figure ----------------------------------------------------
    order = sorted(results, key=lambda k: results[k][0])
    aucs = [results[k][0] for k in order]
    cols = [GREEN if results[k][1] else "#f58231" for k in order]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(order)), aucs, color=cols)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
    ax.axvline(best_single, color="#888", ls=":", label=f"best single layer {best_single:.3f}")
    ax.set_xlim(0.5, max(aucs) + 0.02); ax.set_xlabel("AUROC (5-fold out-of-fold)")
    for i, k in enumerate(order):
        ax.text(results[k][0] + 0.001, i, f"{results[k][0]:.3f}", va="center", fontsize=8)
    ax.set_title("Combining representations - PRM800K heldout (n=6000)\n"
                 "green = leakage-free, orange* = score-stacking (mild optimism)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "combine_ranking.png", dpi=130); plt.close(fig)

    # ---- best leakage-free strategy separation histogram -------------------
    leak_free = {k: v for k, v in results.items() if v[1]}
    best = max(leak_free, key=lambda k: leak_free[k][0])
    s = results[best][2]
    fig, a = plt.subplots(figsize=(7, 5))
    a.hist(s[y == 0], bins=50, alpha=0.6, color=GREEN, label="correct")
    a.hist(s[y == 1], bins=50, alpha=0.6, color=RED, label="incorrect")
    a.set_title(f"best leakage-free combo: {best}\nAUROC={results[best][0]:.3f} "
                f"(single-layer baseline {best_single:.3f})")
    a.set_xlabel("out-of-fold score (high = incorrect)"); a.legend()
    fig.tight_layout(); fig.savefig(ROOT / "combine_best.png", dpi=130); plt.close(fig)

    (ROOT / "repr_combine.json").write_text(json.dumps(
        {"n": n, "layers": layers, "best_single": best_single,
         "best_leakage_free": best,
         "results": {k: {"auc": v[0], "leakage_free": v[1]} for k, v in results.items()}},
        indent=2))
    print(f"\n[best leakage-free] {best}  AUROC={results[best][0]:.4f}  "
          f"(+{results[best][0]-best_single:.3f} over best single layer)")
    print(f"[done] -> {ROOT/'combine_ranking.png'}, {ROOT/'combine_best.png'}")


if __name__ == "__main__":
    main()
