"""S3 first-error analysis: cluster first-error step representations and look at
which ones the probe detects vs misses.

Outputs (under results/s3_first_error/):
  - clustering.html          interactive UMAP scatter of first-error steps,
                             colour=cluster, symbol=detected/missed, hover=step text
  - score_distributions.html interactive score histograms:
                               (a) correct steps vs first-error steps,
                               (b) detected vs missed first-error steps
  - score_distributions.png  static version of the same
  - cluster_summary.csv      per-cluster size / detection-rate / subset mix / mean score

"Detected" = ProcessBench localization hit: the trace's predicted first-error
index equals the gold first-error index (pred_first_error == gold_first_error).
This is exactly what F1_PB rewards. A score-threshold view is also reported.

Usage:
    python scripts/analysis/s3_first_error_clustering.py
    python scripts/analysis/s3_first_error_clustering.py --k_means 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.data.processbench_probe_data import (
    DEFAULT_RUN_DIR,
    SUBSETS,
    load_all,
)


def oracle_thresholds(run_dir: Path) -> dict[str, float]:
    """Per-subset oracle threshold from each shard's metrics.json."""
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

    # ---- masks -----------------------------------------------------------
    fe = d.is_first_error & ~d.skipped
    # "clean" (correct) steps: fully-correct traces, or steps before the error.
    clean = (
        (~d.is_first_error)
        & (~d.skipped)
        & ((d.gold_first_error == -1) | (d.step_idx < d.gold_first_error))
    )
    n_fe = int(fe.sum())
    print(f"[load] {len(d)} steps | {n_fe} first-error steps | {int(clean.sum())} clean steps")

    # Detection: ProcessBench localization hit (pred index == gold index).
    detected = d.pred_first_error == d.gold_first_error
    thr_vec = np.array([thr[s] for s in d.subset])
    score_detected = d.score > thr_vec  # threshold-only view

    H = d.hidden[fe]
    sub_fe = d.subset[fe]
    det_fe = detected[fe]
    scoredet_fe = score_detected[fe]
    score_fe = d.score[fe]

    # ---- embedding + clustering -----------------------------------------
    print("[embed] StandardScaler -> PCA -> UMAP ...")
    Xs = StandardScaler().fit_transform(H)
    Xp = PCA(n_components=min(args.pca, H.shape[1]), random_state=args.seed).fit_transform(Xs)
    emb = UMAP(n_components=2, random_state=args.seed, metric="cosine").fit_transform(Xp)

    print("[cluster] HDBSCAN + KMeans on PCA space ...")
    hdb = HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(Xp)
    km = KMeans(n_clusters=args.k_means, random_state=args.seed, n_init=10).fit_predict(Xp)

    # ---- per-cluster summary (KMeans; stable k) -------------------------
    rows = []
    for c in range(args.k_means):
        m = km == c
        sub_counts = {s: int((sub_fe[m] == s).sum()) for s in SUBSETS}
        rows.append({
            "cluster": c,
            "n": int(m.sum()),
            "detection_rate": round(float(det_fe[m].mean()), 3),
            "mean_score": round(float(score_fe[m].mean()), 3),
            "dominant_subset": max(sub_counts, key=sub_counts.get),
            **{f"n_{s}": sub_counts[s] for s in SUBSETS},
        })
    import csv
    with (args.out_dir / "cluster_summary.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"\n[summary] KMeans k={args.k_means} (overall detection rate={det_fe.mean():.3f})")
    for r in rows:
        print(f"  c{r['cluster']}: n={r['n']:4d}  det={r['detection_rate']:.2f}  "
              f"mean_score={r['mean_score']:.2f}  dom={r['dominant_subset']}")

    # ---- interactive scatter --------------------------------------------
    hover = [
        f"<b>{sub_fe[i]}</b> {d.trace_id[fe][i]} step {int(d.step_idx[fe][i])}"
        f"<br>score={score_fe[i]:.3f} thr={thr[sub_fe[i]]:.3f} "
        f"detected={'Y' if det_fe[i] else 'N'}"
        f"<br>km={km[i]} hdb={hdb[i]}"
        f"<br>---<br>{wrap(d.step_text[fe.nonzero()[0][i]])}"
        for i in range(n_fe)
    ]
    fig = go.Figure()
    palette = pio.templates["plotly"].layout.colorway
    for c in sorted(set(km)):
        for det_flag, sym in [(True, "circle"), (False, "x")]:
            m = (km == c) & (det_fe == det_flag)
            if not m.any():
                continue
            fig.add_trace(go.Scattergl(
                x=emb[m, 0], y=emb[m, 1], mode="markers",
                name=f"c{c} {'det' if det_flag else 'miss'}",
                legendgroup=f"c{c}",
                marker=dict(size=6, symbol=sym,
                            color=palette[c % len(palette)],
                            line=dict(width=0.5, color="white")),
                text=[hover[i] for i in np.where(m)[0]],
                hoverinfo="text",
            ))
    fig.update_layout(
        title=f"First-error step representations (7B, n={n_fe}) "
              f"- UMAP(cosine) of PCA{args.pca}; symbol = detected(o)/missed(x)",
        width=1200, height=800, template="plotly_white",
    )
    fig.write_html(args.out_dir / "clustering.html", include_plotlyjs="cdn")

    # ---- score distributions --------------------------------------------
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=(
        "Probe score: correct steps vs first-error steps",
        "First-error steps: detected vs missed",
    ))
    bins = dict(start=0, end=1, size=0.025)
    fig2.add_trace(go.Histogram(x=d.score[clean], name="correct steps",
                                histnorm="probability density", xbins=bins,
                                marker_color="#2c7fb8", opacity=0.6), 1, 1)
    fig2.add_trace(go.Histogram(x=score_fe, name="first-error steps",
                                histnorm="probability density", xbins=bins,
                                marker_color="#d95f0e", opacity=0.6), 1, 1)
    fig2.add_trace(go.Histogram(x=score_fe[det_fe], name="detected",
                                histnorm="probability density", xbins=bins,
                                marker_color="#1a9850", opacity=0.6), 1, 2)
    fig2.add_trace(go.Histogram(x=score_fe[~det_fe], name="missed",
                                histnorm="probability density", xbins=bins,
                                marker_color="#762a83", opacity=0.6), 1, 2)
    fig2.update_layout(barmode="overlay", template="plotly_white",
                       width=1200, height=500,
                       title="Probe score distributions (7B ProcessBench)")
    fig2.update_xaxes(title_text="probe score P(first-error)")
    fig2.write_html(args.out_dir / "score_distributions.html", include_plotlyjs="cdn")

    # static png
    import matplotlib.pyplot as plt
    fig3, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(d.score[clean], bins=40, density=True, alpha=0.6, label="correct steps")
    ax[0].hist(score_fe, bins=40, density=True, alpha=0.6, label="first-error steps")
    ax[0].set_title("Score: correct vs first-error"); ax[0].legend()
    ax[1].hist(score_fe[det_fe], bins=40, density=True, alpha=0.6, label="detected")
    ax[1].hist(score_fe[~det_fe], bins=40, density=True, alpha=0.6, label="missed")
    ax[1].set_title("First-error: detected vs missed"); ax[1].legend()
    for a in ax:
        a.set_xlabel("probe score P(first-error)")
    fig3.tight_layout()
    fig3.savefig(args.out_dir / "score_distributions.png", dpi=130)

    # ---- headline numbers ------------------------------------------------
    print(f"\n[scores] correct steps:     mean={d.score[clean].mean():.3f}  median={np.median(d.score[clean]):.3f}")
    print(f"[scores] first-error steps: mean={score_fe.mean():.3f}  median={np.median(score_fe):.3f}")
    print(f"[scores] detected first-err mean={score_fe[det_fe].mean():.3f}  "
          f"missed first-err mean={score_fe[~det_fe].mean():.3f}")
    print(f"[detect] localization hit rate (pred==gold) = {det_fe.mean():.3f}")
    print(f"[detect] score>oracle_thr rate              = {scoredet_fe.mean():.3f}")
    print(f"[hdbscan] {len(set(hdb)) - (1 if -1 in hdb else 0)} clusters "
          f"+ {int((hdb == -1).sum())} noise points")
    print(f"\n[done] wrote -> {args.out_dir}/")
    print("  open results/s3_first_error/clustering.html in a browser")


if __name__ == "__main__":
    main()
