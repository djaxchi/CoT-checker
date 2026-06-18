"""Project ALL ProcessBench step encodings onto a 2D map and hunt for clusters.

Unlike the tagged-subset scripts, this uses every step in the run (all ~25.7k
step hidden states, not just first-error steps), embeds them with UMAP on the
FULL 3,584 dims (no PCA bottleneck), then runs HDBSCAN on the embedding to see
whether any clusters fall out on their own.

For every discovered cluster we report what it is made of, using only the
metadata we have for all steps (no failure tags exist outside the 200):
  - subset mix          (is the cluster just a topic / dataset?)
  - first-error rate     (does it concentrate the actual error steps?)
  - mean probe score     (does the probe's signal have spatial structure?)
  - step-position stats  (early vs late steps)

Space:
  --space raw            standardized hidden states (default; shows natural
                         structure, which is usually topic-dominated)
  --space decorrelated   per-subset mean removed (topic whitened out)

Outputs (results/s3_first_error/):
  - projection_all.html              UMAP, dropdown colours {subset, first-error,
                                     probe score, discovered cluster}
  - projection_all_clusters.csv      per-cluster composition

Usage:
    python scripts/analysis/s3_project_all.py
    python scripts/analysis/s3_project_all.py --space decorrelated --min_cluster_size 300
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler, normalize
from umap import UMAP

from src.data.processbench_probe_data import DEFAULT_RUN_DIR, SUBSETS, load_all

ROOT = Path("results/s3_first_error")
PALETTE = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
           "#f032e6", "#bfef45", "#fabed4", "#469990", "#9A6324", "#800000",
           "#808000", "#000075", "#a9a9a9"]
SUBSET_COLOR = {"gsm8k": "#1b9e77", "math": "#d95f02",
                "olympiadbench": "#7570b3", "omnimath": "#e7298a"}


def wrap(text: str, width: int = 90, max_chars: int = 220) -> str:
    text = (text or "")[:max_chars].replace("\n", " ")
    return "<br>".join(text[i : i + width] for i in range(0, len(text), width))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--space", choices=["raw", "decorrelated"], default="raw")
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    ap.add_argument("--min_cluster_size", type=int, default=250)
    ap.add_argument("--n_neighbors", type=int, default=30)
    ap.add_argument("--min_dist", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] reading all step encodings ...")
    d = load_all(args.run_dir, with_text=True)
    ok = ~d.skipped
    idx = np.where(ok)[0]
    n = len(idx)
    H = d.hidden[ok].astype(np.float32)
    sub = d.subset[ok]
    is_fe = d.is_first_error[ok]
    score = d.score[ok]
    sidx = d.step_idx[ok].astype(int)
    nsteps = d.n_steps[ok].astype(int)
    text = [d.step_text[i] for i in idx]
    print(f"[load] {n} steps (first-error: {int(is_fe.sum())}, "
          f"correct/other: {int((~is_fe).sum())})")

    # ---- feature space (full dims) --------------------------------------
    print(f"[prep] standardize ({args.space}) ...")
    X = StandardScaler().fit_transform(H).astype(np.float32)
    if args.space == "decorrelated":
        for s in SUBSETS:
            m = sub == s
            if m.any():
                X[m] -= X[m].mean(axis=0, keepdims=True)
    X = normalize(X)  # unit length -> cosine geometry

    # ---- embed on FULL dims ---------------------------------------------
    print(f"[embed] UMAP on full {X.shape[1]} dims, n={n} (this takes a few min) ...")
    emb = UMAP(n_components=2, random_state=args.seed, metric="cosine",
               n_neighbors=args.n_neighbors, min_dist=args.min_dist).fit_transform(X)

    # ---- find clusters on the map ---------------------------------------
    print(f"[cluster] HDBSCAN(min_cluster_size={args.min_cluster_size}) on embedding ...")
    cl = HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(emb)
    labels_sorted = sorted(set(cl))
    n_clusters = len(labels_sorted) - (1 if -1 in cl else 0)
    print(f"[cluster] {n_clusters} clusters + {int((cl == -1).sum())} noise")

    # ---- per-cluster composition ----------------------------------------
    rows = []
    print(f"\n[clusters] (overall first-error rate={is_fe.mean():.3f}, "
          f"mean score={score.mean():.3f}):")
    for c in labels_sorted:
        m = cl == c
        nc = int(m.sum())
        sub_mix = Counter(sub[m])
        purity = max(sub_mix.values()) / nc
        rows.append({
            "cluster": ("noise" if c == -1 else c), "n": nc,
            "first_error_rate": round(float(is_fe[m].mean()), 3),
            "mean_score": round(float(score[m].mean()), 3),
            "mean_step_frac": round(float((sidx[m] / np.maximum(nsteps[m] - 1, 1)).mean()), 3),
            "dominant_subset": max(sub_mix, key=sub_mix.get),
            "subset_purity": round(purity, 3),
            "subset_mix": "; ".join(f"{s}:{cnt}" for s, cnt in sub_mix.most_common()),
        })
        tag = "noise" if c == -1 else f"c{c}"
        print(f"  {tag:6s} n={nc:5d}  fe_rate={rows[-1]['first_error_rate']:.3f}  "
              f"score={rows[-1]['mean_score']:.3f}  step_frac={rows[-1]['mean_step_frac']:.2f}  "
              f"purity={purity:.2f} dom={rows[-1]['dominant_subset']}")

    with (args.out_dir / "projection_all_clusters.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- interactive scatter with colour-by dropdown --------------------
    hover = [
        f"{sub[i]} step {sidx[i]}/{nsteps[i]-1} "
        f"{'[FIRST-ERROR]' if is_fe[i] else ''}<br>score={score[i]:.3f} "
        f"cluster={'noise' if cl[i]==-1 else cl[i]}<br>{wrap(text[i])}"
        for i in range(n)
    ]
    fig = go.Figure()
    groups: list[tuple[str, list[int]]] = []

    def add_categorical(label, cats, cat_of_point, color_of, namer=str):
        idxs = []
        for ci, cat in enumerate(cats):
            m = cat_of_point == cat
            if not m.any():
                continue
            fig.add_trace(go.Scattergl(
                x=emb[m, 0], y=emb[m, 1], mode="markers", name=namer(cat),
                marker=dict(size=3, color=color_of(cat, ci), opacity=0.55),
                text=[hover[j] for j in np.where(m)[0]], hoverinfo="text",
                visible=False))
            idxs.append(len(fig.data) - 1)
        groups.append((label, idxs))

    # by subset
    add_categorical("by subset", list(SUBSETS), sub,
                    lambda c, i: SUBSET_COLOR.get(c, "#888"))
    # by first-error vs other
    add_categorical("by first-error", [False, True], is_fe,
                    lambda c, i: "#d62728" if c else "#cccccc",
                    namer=lambda c: "first-error" if c else "correct/other")
    # by discovered cluster
    add_categorical("by cluster", labels_sorted, cl,
                    lambda c, i: "#dddddd" if c == -1 else PALETTE[i % len(PALETTE)],
                    namer=lambda c: "noise" if c == -1 else f"c{c}")
    # by probe score (continuous) - single trace
    fig.add_trace(go.Scattergl(
        x=emb[:, 0], y=emb[:, 1], mode="markers", name="probe score",
        marker=dict(size=3, color=score, colorscale="Viridis", opacity=0.6,
                    colorbar=dict(title="score"), showscale=True),
        text=hover, hoverinfo="text", visible=False))
    groups.append(("by probe score", [len(fig.data) - 1]))

    for ti in groups[0][1]:
        fig.data[ti].visible = True

    buttons = []
    for label, idxs in groups:
        vis = [False] * len(fig.data)
        for ti in idxs:
            vis[ti] = True
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": vis},
                                  {"title": f"All ProcessBench steps - coloured {label}"}]))
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor="left",
                          y=1.12, yanchor="top")],
        title=f"All ProcessBench step encodings ({args.space}, full {X.shape[1]} dims; "
              f"n={n}; {n_clusters} HDBSCAN clusters)",
        width=1200, height=820, template="plotly_white")
    out = args.out_dir / "projection_all.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\n[done] wrote -> {out}  +  projection_all_clusters.csv")


if __name__ == "__main__":
    main()
