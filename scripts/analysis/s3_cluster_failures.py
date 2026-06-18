"""Data-driven failure clustering: let the activations define clusters, then read
what kinds of errors land together.

Our failure-mode tags are an imposed taxonomy and may not be the true axes of
incorrectness. So this script does NOT cluster on the tags. It clusters the 200
tagged first-error steps purely on their 7B hidden states (full 3,584 dims,
subset-decorrelated so clusters are not just "which dataset", L2-normalized so
KMeans approximates cosine), then uses our tags + the raw step text only to
*interpret* each cluster.

k is chosen by silhouette (cosine) over a small sweep unless --k is given.

For each discovered cluster we report:
  - size, detection rate, mean probe score, subset mix
  - tag composition + enrichment (tag rate in cluster / tag rate overall), so a
    tag that is *characteristic* of the cluster stands out even if not the mode
  - the steps closest to the centroid (most representative), with text + rationale

Outputs (results/s3_first_error/):
  - cluster_report.html         readable per-cluster cards (MathJax) for reading
  - cluster_failure_composition.csv
  - cluster_scatter.html        UMAP coloured by discovered cluster

Usage:
    python scripts/analysis/s3_cluster_failures.py            # auto-k
    python scripts/analysis/s3_cluster_failures.py --k 6
    python scripts/analysis/s3_cluster_failures.py --space raw
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize
from umap import UMAP

from src.data.processbench_probe_data import DEFAULT_RUN_DIR, SUBSETS, load_all
from src.eval.failure_taxonomy import FAILURE_MODES

ROOT = Path("results/s3_first_error")
PALETTE = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#42d4f4", "#f032e6", "#bfef45", "#a9a9a9", "#469990", "#9A6324"]


def wrap(text: str, width: int = 95, max_chars: int = 700) -> str:
    text = (text or "")[:max_chars].replace("\n", " ")
    return "<br>".join(text[i : i + width] for i in range(0, len(text), width))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--labels", default="failure_labels_final.jsonl")
    ap.add_argument("--space", choices=["decorrelated", "raw"], default="decorrelated")
    ap.add_argument("--k", type=int, default=0, help="0 = pick by silhouette")
    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=9)
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels = {json.loads(l)["sample_id"]: json.loads(l)
              for l in (ROOT / args.labels).read_text().splitlines() if l}
    print(f"[load] {len(labels)} tagged steps; reading hidden states ...")
    d = load_all(args.run_dir, with_text=True)
    fe = d.is_first_error & ~d.skipped
    idx_fe = np.where(fe)[0]
    sid_of = {i: f"{d.trace_id[i]}#s{int(d.step_idx[i])}" for i in idx_fe}
    keep = np.array([i for i in idx_fe if sid_of[i] in labels])
    print(f"[join] matched {len(keep)}/{len(labels)} steps")

    H = d.hidden[keep]
    sub = d.subset[keep]
    detected = (d.pred_first_error == d.gold_first_error)[keep]
    score = d.score[keep]
    text = [d.step_text[i] for i in keep]
    mode = np.array([labels[sid_of[i]]["failure_mode"] for i in keep])
    rationale = [labels[sid_of[i]].get("rationale", "") for i in keep]
    sids = [sid_of[i] for i in keep]
    n = len(keep)

    # ---- build the clustering space (no tags involved) -------------------
    Xs = StandardScaler().fit_transform(H)
    if args.space == "decorrelated":
        for s in SUBSETS:
            m = sub == s
            if m.any():
                Xs[m] -= Xs[m].mean(axis=0, keepdims=True)
    Xn = normalize(Xs)  # unit length -> KMeans ~ spherical (cosine)

    # ---- choose k by silhouette (cosine) --------------------------------
    if args.k and args.k > 0:
        k = args.k
        print(f"[k] using --k {k}")
    else:
        best = (-1.0, args.k_min)
        print("[k] silhouette sweep (cosine):")
        for kk in range(args.k_min, args.k_max + 1):
            lab = KMeans(kk, random_state=args.seed, n_init=10).fit_predict(Xn)
            sil = silhouette_score(Xn, lab, metric="cosine")
            print(f"    k={kk}  silhouette={sil:.3f}")
            if sil > best[0]:
                best = (sil, kk)
        k = best[1]
        print(f"[k] picked k={k} (silhouette={best[0]:.3f})")

    km = KMeans(k, random_state=args.seed, n_init=10)
    cl = km.fit_predict(Xn)

    # ---- per-cluster composition + enrichment ---------------------------
    overall_tag = Counter(mode)
    emb = UMAP(n_components=2, random_state=args.seed, metric="cosine",
               n_neighbors=15, min_dist=0.1).fit_transform(Xn)

    rows = []
    cards = []
    print(f"\n[clusters] k={k} on {args.space} full-dim space "
          f"(overall detection={detected.mean():.2f}):")
    for c in range(k):
        m = cl == c
        nc = int(m.sum())
        if nc == 0:
            continue
        tags = Counter(mode[m])
        # enrichment: P(tag | cluster) / P(tag overall); only for tags present
        enr = {t: (tags[t] / nc) / (overall_tag[t] / n) for t in tags}
        top_tags = tags.most_common(3)
        top_enr = sorted(enr.items(), key=lambda x: -x[1])[:3]
        sub_mix = Counter(sub[m])
        det_rate = float(detected[m].mean())
        ms = float(score[m].mean())
        rows.append({
            "cluster": c, "n": nc, "detection_rate": round(det_rate, 3),
            "mean_score": round(ms, 3),
            "top_tags": "; ".join(f"{t}:{cnt}" for t, cnt in top_tags),
            "enriched_tags": "; ".join(f"{t} x{e:.1f}" for t, e in top_enr),
            "subset_mix": "; ".join(f"{s}:{cnt}" for s, cnt in sub_mix.most_common()),
        })
        print(f"  c{c}: n={nc:3d}  det={det_rate:.2f}  score={ms:.2f}  "
              f"top={rows[-1]['top_tags']}")
        print(f"       enriched: {rows[-1]['enriched_tags']}  | subsets: {rows[-1]['subset_mix']}")

        # representative examples = closest to centroid
        center = Xn[m].mean(0, keepdims=True)
        d2 = ((Xn[m] - center) ** 2).sum(1)
        order = np.where(m)[0][np.argsort(d2)][: args.examples]
        ex = "".join(
            f"<div class=ex><span class=tag>{html.escape(mode[j])}</span> "
            f"<span class=meta>{sids[j]} ({sub[j]}) "
            f"{'DET' if detected[j] else 'MISS'} score={score[j]:.2f}</span><br>"
            f"<i>{html.escape((rationale[j] or '')[:240])}</i><br>{wrap(text[j])}</div>"
            for j in order
        )
        tag_hist = "".join(
            f"<span class=chip>{html.escape(t)}: {cnt}"
            f" <b>(x{enr[t]:.1f})</b></span>"
            for t, cnt in tags.most_common()
        )
        cards.append(f"""
<div class=card style="border-left:6px solid {PALETTE[c % len(PALETTE)]}">
  <div class=hdr>Cluster {c} &middot; n={nc} &middot; detection {det_rate:.0%}
    &middot; mean probe score {ms:.2f}
    &middot; subsets [{rows[-1]['subset_mix']}]</div>
  <div class=tags>{tag_hist}</div>
  <div class=exh>Representative steps (closest to centroid):</div>
  {ex}
</div>""")

    with (ROOT / "cluster_failure_composition.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- readable report ------------------------------------------------
    doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>S3 data-driven failure clusters (k={k}, {args.space})</title>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],
 displayMath:[['$$','$$'],['\\\\[','\\\\]']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
body{{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}
.card{{border:1px solid #ddd;border-radius:8px;padding:1rem 1.2rem;margin:1.2rem 0;background:#fafafa}}
.hdr{{font-size:14px;color:#444;font-weight:600;margin-bottom:.5rem}}
.tags{{margin:.4rem 0 .8rem}}
.chip{{display:inline-block;background:#eef;border-radius:12px;padding:1px 9px;margin:2px;font-size:13px}}
.exh{{font-size:13px;color:#666;margin:.4rem 0}}
.ex{{background:#fff;border:1px solid #eee;border-radius:6px;padding:.5rem .7rem;margin:.4rem 0;font-size:14px}}
.tag{{background:#fff3cd;border-radius:4px;padding:0 6px;font-weight:600;font-size:12px}}
.meta{{color:#888;font-size:12px}}
</style></head><body>
<h2>Data-driven failure clusters &mdash; k={k}, {args.space} space, full 3,584 dims</h2>
<p>Clusters are from the activations alone (tags not used). Tag chips show count and
<b>(x enrichment)</b> = how over-represented that tag is vs the whole sample;
x&gt;1 means characteristic of this cluster.</p>
{''.join(cards)}
</body></html>"""
    (ROOT / "cluster_report.html").write_text(doc)

    # ---- UMAP coloured by discovered cluster ----------------------------
    fig = go.Figure()
    for c in range(k):
        for det_flag, sym in [(True, "circle"), (False, "x")]:
            m = (cl == c) & (detected == det_flag)
            if not m.any():
                continue
            hover = [f"<b>c{c}</b> {sids[j]} ({sub[j]})<br>tag={mode[j]} "
                     f"score={score[j]:.2f}<br>{wrap(text[j], max_chars=400)}"
                     for j in np.where(m)[0]]
            fig.add_trace(go.Scattergl(
                x=emb[m, 0], y=emb[m, 1], mode="markers",
                name=f"c{c} {'det' if det_flag else 'miss'}",
                legendgroup=str(c), showlegend=det_flag,
                marker=dict(size=9, symbol=sym, color=PALETTE[c % len(PALETTE)],
                            line=dict(width=0.5, color="white")),
                text=hover, hoverinfo="text"))
    fig.update_layout(
        title=f"Discovered clusters (k={k}, {args.space}, full-dim; o=detected x=missed; n={n})",
        width=1100, height=780, template="plotly_white")
    fig.write_html(ROOT / "cluster_scatter.html", include_plotlyjs="cdn")

    print(f"\n[done] wrote cluster_report.html + cluster_scatter.html + "
          f"cluster_failure_composition.csv -> {ROOT}/")


if __name__ == "__main__":
    main()
