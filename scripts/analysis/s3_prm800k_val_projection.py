"""Project the PRM800K val_1k encodings onto a 2D map and hunt for clusters.

In-distribution counterpart to scripts/analysis/s3_project_all.py (which mapped the
out-of-distribution ProcessBench steps). PRM800K val_1k is the split the dense
probe's threshold was selected on, so if correctness structure exists anywhere it
should be easiest to see here.

Embeds all ~1,000 val steps on the FULL 3,584 dims (no PCA bottleneck) with UMAP
(cosine), runs HDBSCAN on the map, and reports per-cluster composition using the
labels we actually have for this set: binary step-correctness, PRM800K rating, and
the deployed 7B probe's own score (the probe lives in this exact 3,584-dim space).

Also reports label-decodability: 5-fold balanced logistic-regression accuracy of
predicting step-correctness from the full hidden state, vs the majority floor — the
direct in-distribution comparison to the ProcessBench result.

Outputs (results/prm800k_val/):
  - projection_val.html           UMAP, dropdown colours {label, rating, probe
                                  score, discovered cluster}
  - projection_val_clusters.csv   per-cluster composition

Usage:
    python scripts/analysis/s3_prm800k_val_projection.py
    python scripts/analysis/s3_prm800k_val_projection.py --min_cluster_size 40
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import HDBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, normalize
from umap import UMAP

from src.data.prm800k_val_data import DEFAULT_MERGED_DIR, load_prm800k_val
from src.data.processbench_probe_data import DEFAULT_RUN_DIR, compute_scores, load_probe

ROOT = Path("results/prm800k_val")
PALETTE = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
           "#f032e6", "#bfef45", "#fabed4", "#469990", "#9A6324", "#800000",
           "#808000", "#000075", "#a9a9a9"]
LABEL_COLOR = {0: "#3cb44b", 1: "#e6194B"}          # 0 = correct, 1 = incorrect
RATING_COLOR = {1: "#1a9850", 0: "#999999", -1: "#762a83"}


def classifiability(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """5-fold balanced logreg accuracy of predicting y from X, and majority floor."""
    _, counts = np.unique(y, return_counts=True)
    floor = counts.max() / len(y)
    if len(set(y)) < 2:
        return float("nan"), floor
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    acc = float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean())
    return acc, floor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", type=Path, default=DEFAULT_MERGED_DIR)
    ap.add_argument("--stem", type=str, default="val_1k",
                    help="encoding stem to load, e.g. val_1k or prm800k_heldout_test_6k")
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR,
                    help="run dir holding linear_probe.pt for probe scoring")
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    ap.add_argument("--min_cluster_size", type=int, default=30)
    ap.add_argument("--n_neighbors", type=int, default=20)
    ap.add_argument("--min_dist", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] PRM800K '{args.stem}' from {args.merged_dir} ...")
    d = load_prm800k_val(args.merged_dir, stem=args.stem)
    n = len(d)
    H = d.hidden
    label = d.label
    rating = d.rating
    print(f"[load] {n} steps | correct={int((label==0).sum())} "
          f"incorrect={int((label==1).sum())} | hidden={H.shape[1]}")

    # ---- deployed-probe score (same 3,584-dim space) --------------------
    score = None
    try:
        w, b = load_probe(args.run_dir)
        if w.shape[0] == H.shape[1]:
            score = compute_scores(H, w, b)
            print(f"[probe] scored with {args.run_dir}/linear_probe.pt "
                  f"(mean={score.mean():.3f})")
        else:
            print(f"[probe] dim mismatch ({w.shape[0]} vs {H.shape[1]}); skipping score")
    except Exception as e:  # noqa: BLE001
        print(f"[probe] no probe score ({e})")

    # ---- feature space (full dims, cosine) -----------------------------
    Xs = StandardScaler().fit_transform(H).astype(np.float32)
    Xn = normalize(Xs)

    print(f"[embed] UMAP on full {H.shape[1]} dims, n={n} ...")
    emb = UMAP(n_components=2, random_state=args.seed, metric="cosine",
               n_neighbors=args.n_neighbors, min_dist=args.min_dist).fit_transform(Xn)

    print(f"[cluster] HDBSCAN(min_cluster_size={args.min_cluster_size}) ...")
    cl = HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(emb)
    labels_sorted = sorted(set(cl))
    n_clusters = len(labels_sorted) - (1 if -1 in cl else 0)
    print(f"[cluster] {n_clusters} clusters + {int((cl == -1).sum())} noise")

    # ---- label-decodability (the headline number) ----------------------
    acc, floor = classifiability(Xs, label)
    print(f"\n[decodability] predict step-correctness from full {H.shape[1]} dims "
          f"(5-fold balanced logreg):")
    print(f"  accuracy       = {acc:.3f}")
    print(f"  majority floor = {floor:.3f}")

    # ---- per-cluster composition ---------------------------------------
    rows = []
    print(f"\n[clusters] (overall correct-rate={ (label==0).mean():.3f}):")
    for c in labels_sorted:
        m = cl == c
        nc = int(m.sum())
        row = {
            "cluster": ("noise" if c == -1 else c), "n": nc,
            "correct_rate": round(float((label[m] == 0).mean()), 3),
            "mean_rating": round(float(rating[m].mean()), 3),
            "mean_step_idx": round(float(d.step_idx[m].mean()), 2),
        }
        if score is not None:
            row["mean_probe_score"] = round(float(score[m].mean()), 3)
        rows.append(row)
        tag = "noise" if c == -1 else f"c{c}"
        extra = f" score={row.get('mean_probe_score','-')}" if score is not None else ""
        print(f"  {tag:6s} n={nc:4d}  correct={row['correct_rate']:.3f}  "
              f"rating={row['mean_rating']:+.2f}{extra}")

    with (args.out_dir / "projection_val_clusters.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- interactive scatter with colour-by dropdown -------------------
    hover = [
        f"{d.uid[i]} | prob={d.problem_id[i]} step={d.step_idx[i]}"
        f"<br>label={'correct' if label[i]==0 else 'incorrect'} rating={rating[i]:+d}"
        f"{'' if score is None else f' | probe={score[i]:.3f}'}"
        f"<br>cluster={'noise' if cl[i]==-1 else cl[i]}"
        for i in range(n)
    ]
    fig = go.Figure()
    groups: list[tuple[str, list[int]]] = []

    def add_categorical(label_name, cats, cat_of_point, color_of, namer=str):
        idxs = []
        for ci, cat in enumerate(cats):
            m = cat_of_point == cat
            if not m.any():
                continue
            fig.add_trace(go.Scattergl(
                x=emb[m, 0], y=emb[m, 1], mode="markers", name=namer(cat),
                marker=dict(size=5, color=color_of(cat, ci), opacity=0.7),
                text=[hover[j] for j in np.where(m)[0]], hoverinfo="text",
                visible=False))
            idxs.append(len(fig.data) - 1)
        groups.append((label_name, idxs))

    add_categorical("by label", [0, 1], label,
                    lambda c, i: LABEL_COLOR[c],
                    namer=lambda c: "correct" if c == 0 else "incorrect")
    add_categorical("by rating", sorted(set(rating.tolist())), rating,
                    lambda c, i: RATING_COLOR.get(c, PALETTE[i % len(PALETTE)]),
                    namer=lambda c: f"rating {c:+d}")
    add_categorical("by cluster", labels_sorted, cl,
                    lambda c, i: "#dddddd" if c == -1 else PALETTE[i % len(PALETTE)],
                    namer=lambda c: "noise" if c == -1 else f"c{c}")
    if score is not None:
        fig.add_trace(go.Scattergl(
            x=emb[:, 0], y=emb[:, 1], mode="markers", name="probe score",
            marker=dict(size=5, color=score, colorscale="Viridis", opacity=0.75,
                        colorbar=dict(title="score"), showscale=True),
            text=hover, hoverinfo="text", visible=False))
        groups.append(("by probe score", [len(fig.data) - 1]))

    for ti in groups[0][1]:
        fig.data[ti].visible = True

    buttons = []
    for label_name, idxs in groups:
        vis = [False] * len(fig.data)
        for ti in idxs:
            vis[ti] = True
        buttons.append(dict(label=label_name, method="update",
                            args=[{"visible": vis},
                                  {"title": f"PRM800K val_1k - coloured {label_name}"}]))
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor="left",
                          y=1.12, yanchor="top")],
        title=f"PRM800K val_1k step encodings (7B, full {H.shape[1]} dims; n={n}; "
              f"{n_clusters} HDBSCAN clusters; decodability {acc:.2f} vs floor {floor:.2f})",
        width=1150, height=820, template="plotly_white")
    out = args.out_dir / "projection_val.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\n[done] wrote -> {out}  +  projection_val_clusters.csv")


if __name__ == "__main__":
    main()
