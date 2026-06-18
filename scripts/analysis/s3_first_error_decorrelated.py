"""S3 first-error analysis, subset-decorrelated.

Raw 7B hidden geometry is dominated by topic/subset. This script removes the
per-subset mean (after standardizing), re-embeds and re-clusters, and asks:
once topic is whitened out, do failure-mode clusters appear, and is detection
explained by anything beyond subset?

Quantifies the confound with a subset-classifiability probe (cross-val logistic
regression accuracy on the PCA features) BEFORE vs AFTER decorrelation: if it
falls toward the majority-class baseline, topic structure was removed.

Output (results/s3_first_error/):
  - decorrelated_clustering.html  one interactive UMAP scatter with a dropdown to
                                  colour by {subset, cluster, detected/missed};
                                  symbol always encodes detected(o)/missed(x);
                                  hover shows the step text + score.
  - decorrelated_cluster_summary.csv

Usage:
    python scripts/analysis/s3_first_error_decorrelated.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.data.processbench_probe_data import DEFAULT_RUN_DIR, SUBSETS, load_all


def oracle_thresholds(run_dir: Path) -> dict[str, float]:
    out = {}
    for sub in SUBSETS:
        m = json.loads(
            (run_dir / "processbench_eval_shards" / sub / "metrics.json").read_text()
        )
        out[sub] = float(m["oracle"]["threshold"])
    return out


def wrap(text: str, width: int = 90, max_chars: int = 600) -> str:
    text = text[:max_chars].replace("\n", " ")
    return "<br>".join(text[i : i + width] for i in range(0, len(text), width))


def subset_classifiability(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Mean 5-fold accuracy of predicting subset from features (chance = mixing)."""
    clf = LogisticRegression(max_iter=2000, C=1.0)
    return float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--out_dir", type=Path, default=Path("results/s3_first_error"))
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--k_means", type=int, default=8)
    ap.add_argument("--min_cluster_size", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] reading all subsets ...")
    d = load_all(args.run_dir, with_text=True)
    thr = oracle_thresholds(args.run_dir)

    fe = d.is_first_error & ~d.skipped
    idx_fe = np.where(fe)[0]
    n = len(idx_fe)
    H = d.hidden[fe]
    sub = d.subset[fe]
    detected = (d.pred_first_error == d.gold_first_error)[fe]
    score = d.score[fe]
    text = [d.step_text[i] for i in idx_fe]
    tid = d.trace_id[fe]
    sidx = d.step_idx[fe]
    print(f"[load] {n} first-error steps")

    # ---- standardize, then whiten out per-subset means -------------------
    Xs = StandardScaler().fit_transform(H)
    Xd = Xs.copy()
    for s in SUBSETS:
        m = sub == s
        if m.any():
            Xd[m] -= Xs[m].mean(axis=0, keepdims=True)

    pca_raw = PCA(args.pca, random_state=args.seed).fit_transform(Xs)
    pca_dec = PCA(args.pca, random_state=args.seed).fit_transform(Xd)

    # ---- did we remove topic? -------------------------------------------
    _, counts = np.unique(sub, return_counts=True)
    majority = counts.max() / n
    acc_raw = subset_classifiability(pca_raw, sub, args.seed)
    acc_dec = subset_classifiability(pca_dec, sub, args.seed)
    print("\n[confound] subset-classifiability (5-fold logreg accuracy):")
    print(f"  raw            = {acc_raw:.3f}")
    print(f"  decorrelated   = {acc_dec:.3f}")
    print(f"  majority-class = {majority:.3f}  (floor if topic fully removed)")

    # ---- embed + cluster on the decorrelated space ----------------------
    print("[embed] UMAP(cosine) on decorrelated PCA ...")
    emb = UMAP(n_components=2, random_state=args.seed, metric="cosine").fit_transform(pca_dec)
    km = KMeans(args.k_means, random_state=args.seed, n_init=10).fit_predict(pca_dec)
    hdb = HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(pca_dec)

    # ---- per-cluster summary (purity + detection) -----------------------
    rows = []
    for c in range(args.k_means):
        m = km == c
        sub_counts = {s: int((sub[m] == s).sum()) for s in SUBSETS}
        purity = max(sub_counts.values()) / max(int(m.sum()), 1)
        rows.append({
            "cluster": c, "n": int(m.sum()),
            "detection_rate": round(float(detected[m].mean()), 3),
            "mean_score": round(float(score[m].mean()), 3),
            "subset_purity": round(purity, 3),
            "dominant_subset": max(sub_counts, key=sub_counts.get),
            **{f"n_{s}": sub_counts[s] for s in SUBSETS},
        })
    with (args.out_dir / "decorrelated_cluster_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[clusters] KMeans k={args.k_means} on decorrelated space "
          f"(overall detection={detected.mean():.3f}):")
    for r in rows:
        print(f"  c{r['cluster']}: n={r['n']:4d}  det={r['detection_rate']:.2f}  "
              f"mean_score={r['mean_score']:.2f}  purity={r['subset_purity']:.2f}  "
              f"dom={r['dominant_subset']}")
    print(f"[hdbscan] {len(set(hdb)) - (1 if -1 in hdb else 0)} clusters "
          f"+ {int((hdb == -1).sum())} noise")

    # ---- interactive scatter with colour-by dropdown --------------------
    hover = [
        f"<b>{sub[i]}</b> {tid[i]} step {int(sidx[i])}"
        f"<br>score={score[i]:.3f} thr={thr[sub[i]]:.3f} "
        f"detected={'Y' if detected[i] else 'N'} | km={km[i]}"
        f"<br>---<br>{wrap(text[i])}"
        for i in range(n)
    ]
    palette = pio.templates["plotly"].layout.colorway
    fig = go.Figure()
    groups = []  # (label, list_of_trace_indices)

    def add_group(label, categories, cat_of_point, color_of_cat):
        idxs = []
        for ci, cat in enumerate(categories):
            for det_flag, sym in [(True, "circle"), (False, "x")]:
                m = (cat_of_point == cat) & (detected == det_flag)
                if not m.any():
                    continue
                fig.add_trace(go.Scattergl(
                    x=emb[m, 0], y=emb[m, 1], mode="markers",
                    name=f"{cat} {'det' if det_flag else 'miss'}",
                    marker=dict(size=6, symbol=sym, color=color_of_cat(cat, ci),
                                line=dict(width=0.4, color="white")),
                    text=[hover[i] for i in np.where(m)[0]],
                    hoverinfo="text", visible=False,
                ))
                idxs.append(len(fig.data) - 1)
        groups.append((label, idxs))

    add_group("by subset", list(SUBSETS), sub,
              lambda cat, ci: palette[ci % len(palette)])
    add_group("by cluster", list(range(args.k_means)), km,
              lambda cat, ci: palette[ci % len(palette)])
    add_group("by detected/missed", [True, False], detected,
              lambda cat, ci: "#1a9850" if cat else "#762a83")

    for ti in groups[0][1]:  # default: show "by subset"
        fig.data[ti].visible = True

    buttons = []
    for label, idxs in groups:
        vis = [False] * len(fig.data)
        for ti in idxs:
            vis[ti] = True
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": vis}, {"title": f"First-error steps - coloured {label}"}]))
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor="left",
                          y=1.12, yanchor="top")],
        title="First-error steps - coloured by subset "
              f"(subset-decorrelated; symbol o=detected x=missed; n={n})",
        width=1200, height=820, template="plotly_white",
    )
    fig.write_html(args.out_dir / "decorrelated_clustering.html", include_plotlyjs="cdn")

    print(f"\n[done] wrote -> {args.out_dir}/decorrelated_clustering.html")
    print("  use the dropdown (top-left) to switch colouring")


if __name__ == "__main__":
    main()
