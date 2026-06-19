"""Find the MINIMAL set of last-token activations that separates correct/incorrect,
drop the activations that are common to both classes, and re-map only the survivors.

Follow-up to s3_prm800k_probe_anatomy.py, which showed the correctness signal is a thin
linear margin spread over a low-variance subspace (invisible to a full-dim UMAP). Here we:

  1. score every hidden dim by how differently it behaves across the two classes
     (Welch t / Cohen's d) -> "discriminative" vs "common" dims,
  2. find the minimal predictive subset with an L1 (sparse) logistic probe: sweep C,
     record (#nonzero dims, 5-fold accuracy), take the smallest set within tol of the
     full-dim accuracy. L1 keeps a non-redundant subset and zeroes the common dims,
  3. re-embed with UMAP using ONLY the minimal subspace and ask whether labels now
     fall into clusters, contrasted against (a) the COMMON subspace and (b) a
     SHUFFLED-LABEL control selected the same way (guards against selection inventing
     structure), and an honest train/test split to confirm the set generalises.

Outputs (results/prm800k_layers/minimal_subspace/):
  - minimal_subspace.html     UMAP, dropdown {minimal / common / shuffled control},
                              coloured correct/incorrect.
  - minimal_subspace.png      accuracy-vs-#features curve, per-dim effect sizes,
                              2D-separability real-vs-shuffled, cluster purity.
  - minimal_subspace.json     selected dim indices + every number.

Usage:
    python scripts/analysis/s3_prm800k_minimal_subspace.py            # L20 / last
    python scripts/analysis/s3_prm800k_minimal_subspace.py --layer 28
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.stats import ttest_ind
from sklearn.cluster import HDBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.data.prm800k_val_data import load_prm800k_multitoken

DEFAULT_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b/prm_multitoken")
ROOT = Path("results/prm800k_layers/minimal_subspace")
SEED = 42
LABEL_COLOR = {0: "#3cb44b", 1: "#e6194B"}        # 0 = correct, 1 = incorrect


def l2(C=1.0):
    return LogisticRegression(max_iter=4000, C=C, class_weight="balanced")


def l1(C):
    return LogisticRegression(penalty="l1", solver="liblinear", C=C,
                              max_iter=3000, class_weight="balanced")


def cv_acc(est, X, y):
    return float(cross_val_score(est, X, y, cv=5, scoring="accuracy").mean())


def knn2d(emb, y):
    """How separable are the labels in the 2D map alone (5-fold kNN accuracy)."""
    return float(cross_val_score(KNeighborsClassifier(15), emb, y, cv=5,
                                 scoring="accuracy").mean())


def embed(X, seed=SEED):
    return UMAP(n_components=2, random_state=seed, metric="cosine",
                n_neighbors=20, min_dist=0.0).fit_transform(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--stem", type=str, default="prm800k_heldout_test")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--token", type=str, default="last")
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="accuracy we allow the minimal set to fall below full-dim")
    ap.add_argument("--min_cluster_size", type=int, default=30)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = load_prm800k_multitoken(args.merged_dir, args.stem, args.layer, args.token)
    n = len(d)
    y = d.label.astype(int)                       # 1 = incorrect
    Xs = StandardScaler().fit_transform(d.hidden).astype(np.float32)
    P = Xs.shape[1]
    full_acc = cv_acc(l2(), Xs, y)
    print(f"[plane] L{args.layer}/{args.token}  n={n} dims={P}  full-dim acc={full_acc:.3f}")
    R: dict = {"plane": f"L{args.layer}/{args.token}", "n": n, "dims": P,
               "full_acc": round(full_acc, 4)}

    # ---- per-dim discriminability: which dims differ between classes -----
    t, _ = ttest_ind(Xs[y == 1], Xs[y == 0], axis=0, equal_var=False)
    cohend = Xs[y == 1].mean(0) - Xs[y == 0].mean(0)      # std=1 dims -> ~Cohen's d
    absd = np.abs(cohend)
    order = np.argsort(absd)[::-1]                        # most discriminative first
    print(f"[per-dim] |Cohen d|: max={absd.max():.3f} median={np.median(absd):.3f}  "
          f"dims with |d|>0.2: {(absd>0.2).sum()}  |d|>0.1: {(absd>0.1).sum()}")
    R["per_dim"] = {"max_abs_d": round(float(absd.max()), 3),
                    "median_abs_d": round(float(np.median(absd)), 3),
                    "n_d_gt_0.2": int((absd > 0.2).sum()),
                    "n_d_gt_0.1": int((absd > 0.1).sum())}

    # ---- minimal predictive set via L1 sparse probe ---------------------
    Cs = [0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0]
    l1_curve = []
    print("\n[L1 path] C -> (#nonzero dims, 5-fold acc):")
    for C in Cs:
        acc = cv_acc(l1(C), Xs, y)
        nnz = int((l1(C).fit(Xs, y).coef_ != 0).sum())
        l1_curve.append({"C": C, "nnz": nnz, "acc": round(acc, 4)})
        print(f"  C={C:<6} nnz={nnz:4d}  acc={acc:.3f}")
    R["l1_curve"] = l1_curve
    ok = [r for r in l1_curve if r["acc"] >= full_acc - args.tol]
    pick = min(ok, key=lambda r: r["nnz"]) if ok else max(l1_curve, key=lambda r: r["acc"])
    Cstar = pick["C"]
    coef = l1(Cstar).fit(Xs, y).coef_.ravel()
    S = np.where(coef != 0)[0]
    m = len(S)
    print(f"[minimal] C*={Cstar}  |S|={m} dims  acc={pick['acc']:.3f} "
          f"(full {full_acc:.3f}, floor 0.50)  -> kept {m}/{P} = {100*m/P:.1f}%")
    R["minimal"] = {"C_star": Cstar, "size": m, "frac_kept": round(m / P, 4),
                    "acc": pick["acc"], "dim_indices": S.tolist()}

    # ---- univariate top-k curve (selection by |d|) ----------------------
    ks = [5, 10, 20, 50, 100, 200, 400, 800]
    uni_curve = [{"k": k, "acc": round(cv_acc(l2(), Xs[:, order[:k]], y), 4)} for k in ks]
    R["univariate_curve"] = uni_curve
    print("[univariate] top-k by |d| -> acc:",
          " ".join(f"{r['k']}:{r['acc']:.3f}" for r in uni_curve))

    # ---- honest generalisation: select on train, score on test ----------
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.5, random_state=SEED,
                                          stratify=y)
    Str = np.where(l1(Cstar).fit(Xtr, ytr).coef_.ravel() != 0)[0]
    gen = float((l2().fit(Xtr[:, Str], ytr).predict(Xte[:, Str]) == yte).mean())
    print(f"[generalise] select {len(Str)} dims on train half -> test acc={gen:.3f}")
    R["generalisation"] = {"train_selected": int(len(Str)), "test_acc": round(gen, 4)}

    # ---- define the three subspaces to map ------------------------------
    common = order[-m:]                                   # least discriminative m dims
    yperm = rng.permutation(y)
    tp, _ = ttest_ind(Xs[yperm == 1], Xs[yperm == 0], axis=0, equal_var=False)
    shuf = np.argsort(np.abs(tp))[::-1][:m]               # top-m by chance under shuffle
    spaces = {
        "minimal (discriminative)": (S, y),
        "common (eliminated)": (common, y),
        "shuffled-label control": (shuf, y),
    }

    # ---- map each subspace, test for clusters + 2D separability ---------
    fig = go.Figure()
    groups, sep_rows, cluster_rows = [], [], []
    hover = [f"{d.uid[i]} | prob={d.problem_id[i]} step={d.step_idx[i]}"
             f"<br>{'incorrect' if y[i] else 'correct'} rating={d.rating[i]:+d}"
             for i in range(n)]
    for name, (dims, ycol) in spaces.items():
        emb = embed(Xs[:, dims])
        sep = knn2d(emb, y)
        cl = HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(emb)
        nclu = len(set(cl.tolist())) - (1 if -1 in cl else 0)
        # cluster label purity: how lopsided are clusters by correctness
        purities = [max((y[cl == c] == 0).mean(), (y[cl == c] == 1).mean())
                    for c in set(cl.tolist()) if c != -1]
        mean_pur = float(np.mean(purities)) if purities else float("nan")
        sep_rows.append((name, sep))
        cluster_rows.append((name, nclu, mean_pur))
        print(f"[map] {name:26s} dims={len(dims):4d}  2D-kNN sep={sep:.3f}  "
              f"clusters={nclu}  mean purity={mean_pur:.3f}")
        idxs = []
        for c in (0, 1):
            mk = y == c
            fig.add_trace(go.Scattergl(
                x=emb[mk, 0], y=emb[mk, 1], mode="markers",
                name="correct" if c == 0 else "incorrect",
                marker=dict(size=4, color=LABEL_COLOR[c], opacity=0.6),
                text=[hover[j] for j in np.where(mk)[0]], hoverinfo="text",
                visible=False))
            idxs.append(len(fig.data) - 1)
        groups.append((f"{name}  [2D-sep {sep:.2f}, {nclu} clusters]", idxs))
    R["map_separability"] = {k: round(v, 4) for k, v in sep_rows}
    R["map_clusters"] = {k: {"n_clusters": nc, "mean_purity": round(p, 4)}
                         for k, nc, p in cluster_rows}

    for ti in groups[0][1]:
        fig.data[ti].visible = True
    buttons = []
    for label, idxs in groups:
        vis = [False] * len(fig.data)
        for ti in idxs:
            vis[ti] = True
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": vis},
                                  {"title": f"PRM800K L{args.layer}/{args.token} - {label}"}]))
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor="left",
                          y=1.12, yanchor="top")],
        title=f"PRM800K L{args.layer}/{args.token} - {groups[0][0]}",
        width=1150, height=820, template="plotly_white")
    html = args.out_dir / "minimal_subspace.html"
    fig.write_html(html, include_plotlyjs="cdn")
    (args.out_dir / "minimal_subspace.json").write_text(json.dumps(R, indent=2))

    # ---- summary figure -------------------------------------------------
    fig2, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot([r["nnz"] for r in l1_curve], [r["acc"] for r in l1_curve],
                  "-o", label="L1 sparse probe")
    ax[0, 0].plot([r["k"] for r in uni_curve], [r["acc"] for r in uni_curve],
                  "--s", label="univariate top-k")
    ax[0, 0].axhline(full_acc, color="#444", ls=":", label=f"full {P} dims ({full_acc:.2f})")
    ax[0, 0].axhline(0.5, color="#888", ls=":", label="floor")
    ax[0, 0].axvline(m, color="#911eb4", ls="--", alpha=0.7, label=f"minimal={m}")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("# activations kept")
    ax[0, 0].set_ylabel("5-fold accuracy"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("how few activations still separate the classes")

    ax[0, 1].hist(absd, bins=80, color="#4363d8")
    ax[0, 1].axvline(absd[S].min() if m else 0, color="#911eb4", ls="--",
                     label=f"min |d| in set ({absd[S].min():.2f})" if m else "")
    ax[0, 1].set_xlabel("|Cohen's d| per activation"); ax[0, 1].set_ylabel("count")
    ax[0, 1].set_yscale("log"); ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("most activations are common to both classes")

    names = [r[0].split(" ")[0] for r in sep_rows]
    ax[1, 0].bar(range(len(sep_rows)), [r[1] for r in sep_rows],
                 color=["#911eb4", "#888", "#bbb"])
    ax[1, 0].axhline(0.5, color="#888", ls=":")
    ax[1, 0].set_xticks(range(len(names))); ax[1, 0].set_xticklabels(names)
    ax[1, 0].set_ylim(0.45, max(r[1] for r in sep_rows) + 0.03)
    for i, r in enumerate(sep_rows):
        ax[1, 0].text(i, r[1] + 0.003, f"{r[1]:.3f}", ha="center", fontsize=9)
    ax[1, 0].set_title("2D-map label separability (kNN)\nminimal vs shuffled control")

    ax[1, 1].bar(range(len(cluster_rows)), [r[2] for r in cluster_rows],
                 color=["#911eb4", "#888", "#bbb"])
    ax[1, 1].axhline(0.5, color="#888", ls=":", label="0.5 = no purity")
    ax[1, 1].set_xticks(range(len(names))); ax[1, 1].set_xticklabels(names)
    ax[1, 1].set_ylim(0.45, 1.0); ax[1, 1].legend(fontsize=8)
    for i, r in enumerate(cluster_rows):
        ax[1, 1].text(i, r[2] + 0.005, f"{r[2]:.2f}\n({r[1]}cl)", ha="center", fontsize=8)
    ax[1, 1].set_title("HDBSCAN cluster correctness-purity")

    fig2.suptitle(f"Minimal separating subspace - PRM800K L{args.layer}/{args.token}  "
                  f"(kept {m}/{P} dims, acc {pick['acc']:.2f}, test {gen:.2f})", fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    png = args.out_dir / "minimal_subspace.png"
    fig2.savefig(png, dpi=130); plt.close(fig2)

    print(f"\n[done] -> {html}")
    print(f"        + {png}")
    print(f"        + {args.out_dir / 'minimal_subspace.json'}")


if __name__ == "__main__":
    main()
