"""Anatomy of the PRM800K step-correctness signal: what does the probe actually read?

The layer sweep (s3_prm800k_layer_projection.py) shows ~0.74 linear decodability on the
last-token hidden state, yet the UMAP map has no correctness clusters. So the signal is
linear-but-distributed. This script dissects it on a single (layer, token) plane and,
crucially, separates genuine "content" signal from two confounds that track the label:

    step length (n_tokens)   incorrect steps are longer
    step position (step_idx)  incorrect steps come later

It reports, all 5-fold balanced (accuracy + ROC-AUC):
  - decodability from the confounds ALONE (the floor any hidden-state number must beat),
  - decodability from the full hidden state,
  - the same after L2-normalising rows (kills magnitude/norm  -> pure direction/cosine),
  - the same after linearly regressing the confounds out of every dim (residual signal),
  - how the probe's own score correlates with length / position,
  - how the signal is spread over PCA dimensions (top-k kept, and top-k removed) -> why
    it is invisible to a 2D map,
  - the raw last-token NORM as a one-number classifier.

Outputs (results/prm800k_layers/probe_anatomy/):
  - anatomy_L{layer}_{token}.png    4-panel summary
  - anatomy_L{layer}_{token}.json   every number

Usage:
    python scripts/analysis/s3_prm800k_probe_anatomy.py                 # L20 / last
    python scripts/analysis/s3_prm800k_probe_anatomy.py --layer 28
    python scripts/analysis/s3_prm800k_probe_anatomy.py --token first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler, normalize

from src.data.prm800k_val_data import load_prm800k_multitoken

DEFAULT_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b/prm_multitoken")
ROOT = Path("results/prm800k_layers/probe_anatomy")
SEED = 42


def clf():
    return LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")


def decode(X: np.ndarray, y: np.ndarray) -> dict:
    """5-fold balanced accuracy + ROC-AUC of predicting y from X."""
    X = np.atleast_2d(X.astype(np.float32))
    if X.shape[0] != len(y):
        X = X.T
    acc = float(cross_val_score(clf(), X, y, cv=5, scoring="accuracy").mean())
    auc = float(cross_val_score(clf(), X, y, cv=5, scoring="roc_auc").mean())
    return {"acc": round(acc, 4), "auc": round(auc, 4)}


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
    n = len(d)
    y = d.label.astype(int)                       # 1 = incorrect (rating -1)
    floor = max(np.bincount(y)) / n
    H = d.hidden.astype(np.float32)
    raw_norm = np.linalg.norm(H, axis=1)
    nt = d.n_tokens.astype(np.float32)
    si = d.step_idx.astype(np.float32)
    print(f"[plane] L{args.layer}/{args.token}  n={n}  dims={H.shape[1]}  "
          f"floor={floor:.3f}  (1=incorrect: {int(y.sum())})")

    R: dict = {"plane": f"L{args.layer}/{args.token}", "n": n, "dims": H.shape[1],
               "floor": round(floor, 4)}

    # ---- standardized feature spaces -----------------------------------
    Xs = StandardScaler().fit_transform(H).astype(np.float32)   # per-dim z-score
    Xn = normalize(Xs)                                          # unit rows (cosine)

    # ---- confound baselines (the bar any hidden number must clear) -----
    conf = StandardScaler().fit_transform(np.column_stack([nt, si]))
    R["confounds"] = {
        "n_tokens_only": decode(StandardScaler().fit_transform(nt[:, None]), y),
        "step_idx_only": decode(StandardScaler().fit_transform(si[:, None]), y),
        "n_tokens+step_idx": decode(conf, y),
        "raw_norm_only": decode(StandardScaler().fit_transform(raw_norm[:, None]), y),
    }
    print("\n[confounds] decodability from metadata alone:")
    for k, v in R["confounds"].items():
        print(f"  {k:20s} acc={v['acc']:.3f}  auc={v['auc']:.3f}")
    print(f"  incorrect-step length: median {np.median(nt[y==1]):.0f} vs "
          f"correct {np.median(nt[y==0]):.0f} tokens")

    # ---- hidden-state spaces, with confound controls -------------------
    R["full_standardized"] = decode(Xs, y)
    R["row_normalized_cosine"] = decode(Xn, y)

    # regress confounds (+intercept) out of every standardized dim
    C = np.column_stack([np.ones(n), nt, si]).astype(np.float32)
    beta, *_ = np.linalg.lstsq(C, Xs, rcond=None)
    X_resid = Xs - C @ beta
    R["confound_residualized"] = decode(X_resid, y)

    print("\n[hidden state] decodability under confound controls:")
    print(f"  full standardized        acc={R['full_standardized']['acc']:.3f}  "
          f"auc={R['full_standardized']['auc']:.3f}")
    print(f"  row-normalized (cosine)  acc={R['row_normalized_cosine']['acc']:.3f}  "
          f"auc={R['row_normalized_cosine']['auc']:.3f}   (magnitude removed)")
    print(f"  confound-residualized    acc={R['confound_residualized']['acc']:.3f}  "
          f"auc={R['confound_residualized']['auc']:.3f}   (length+position removed)")

    # ---- probe direction vs confounds ----------------------------------
    score = cross_val_predict(clf(), Xs, y, cv=5, method="decision_function")
    R["probe_score_vs_confound"] = {
        "pearson_n_tokens": round(float(pearsonr(score, nt)[0]), 3),
        "spearman_n_tokens": round(float(spearmanr(score, nt)[0]), 3),
        "pearson_step_idx": round(float(pearsonr(score, si)[0]), 3),
        "pearson_raw_norm": round(float(pearsonr(score, raw_norm)[0]), 3),
    }
    print("\n[probe direction] correlation of the probe score with confounds:")
    for k, v in R["probe_score_vs_confound"].items():
        print(f"  {k:20s} r={v:+.3f}")

    # ---- dimensionality: where does the signal live? -------------------
    pca = PCA(n_components=min(512, H.shape[1]), random_state=SEED).fit(Xs)
    Z = pca.transform(Xs)
    ks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
    keep, remove = [], []
    for k in ks:
        keep.append({"k": k, **decode(Z[:, :k], y)})
        Xr = Xs - Z[:, :k] @ pca.components_[:k]      # strip top-k PCs
        remove.append({"k": k, **decode(Xr, y)})
    R["pca_keep_topk"] = keep
    R["pca_remove_topk"] = remove
    print("\n[dimensionality] keep top-k PCs / remove top-k PCs (acc):")
    for a, b in zip(keep, remove):
        print(f"  k={a['k']:4d}   keep={a['acc']:.3f}   remove-top={b['acc']:.3f}")

    # ---- raw norm as a classifier --------------------------------------
    R["raw_norm_auc"] = R["confounds"]["raw_norm_only"]["auc"]

    (args.out_dir / f"anatomy_L{args.layer}_{args.token}.json").write_text(
        json.dumps(R, indent=2))

    # ---- figure --------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    full_auc = R["full_standardized"]["auc"]

    # (1) the 1D margin the probe actually finds
    ax[0, 0].hist(score[y == 0], bins=50, alpha=0.6, color="#3cb44b", label="correct")
    ax[0, 0].hist(score[y == 1], bins=50, alpha=0.6, color="#e6194B", label="incorrect")
    ax[0, 0].set_title(f"probe's 1D margin (out-of-fold score)  AUC={full_auc:.3f}\n"
                       "the separation UMAP cannot show")
    ax[0, 0].set_xlabel("probe decision score"); ax[0, 0].legend()

    # (2) decodability vs PCA dims
    ax[0, 1].plot(ks, [r["acc"] for r in keep], "-o", label="keep top-k PCs")
    ax[0, 1].plot(ks, [r["acc"] for r in remove], "--s", label="remove top-k PCs")
    ax[0, 1].axhline(R["full_standardized"]["acc"], color="#444", ls=":",
                     label="full dims")
    ax[0, 1].axhline(floor, color="#888", ls=":", label=f"floor {floor:.2f}")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("k (PCA components)")
    ax[0, 1].set_ylabel("decodability"); ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("signal is spread across many low-variance dims")

    # (3) raw norm as a 1-number classifier
    ax[1, 0].hist(raw_norm[y == 0], bins=50, alpha=0.6, color="#3cb44b", label="correct")
    ax[1, 0].hist(raw_norm[y == 1], bins=50, alpha=0.6, color="#e6194B", label="incorrect")
    ax[1, 0].set_title(f"raw last-token norm  AUC={R['raw_norm_auc']:.3f}")
    ax[1, 0].set_xlabel("||hidden state||"); ax[1, 0].legend()

    # (4) confound vs signal bar
    bars = {
        "confounds\n(len+pos)": R["confounds"]["n_tokens+step_idx"]["acc"],
        "raw norm": R["confounds"]["raw_norm_only"]["acc"],
        "full\nhidden": R["full_standardized"]["acc"],
        "cosine\n(no mag)": R["row_normalized_cosine"]["acc"],
        "resid\n(no len/pos)": R["confound_residualized"]["acc"],
    }
    ax[1, 1].bar(range(len(bars)), list(bars.values()),
                 color=["#888", "#bbb", "#4363d8", "#42d4f4", "#911eb4"])
    ax[1, 1].set_xticks(range(len(bars))); ax[1, 1].set_xticklabels(list(bars), fontsize=8)
    ax[1, 1].axhline(floor, color="#888", ls=":")
    ax[1, 1].set_ylim(0.45, max(bars.values()) + 0.03)
    for i, v in enumerate(bars.values()):
        ax[1, 1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax[1, 1].set_title("what survives each control")

    fig.suptitle(f"Probe anatomy - PRM800K {args.stem}  L{args.layer}/{args.token}  "
                 f"(n={n}, floor={floor:.2f})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = args.out_dir / f"anatomy_L{args.layer}_{args.token}.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    print(f"\n[done] -> {png}")
    print(f"        + {args.out_dir / f'anatomy_L{args.layer}_{args.token}.json'}")


if __name__ == "__main__":
    main()
