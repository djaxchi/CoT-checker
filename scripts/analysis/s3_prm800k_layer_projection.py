"""Sweep the layer axis of the PRM800K multi-layer/multi-token encoding and map it.

Multi-layer counterpart to scripts/analysis/s3_prm800k_val_projection.py and
s3_project_all.py. Those two map a single (layer, token) plane; this one walks every
plane stored in the 4D encoding (n, L, T, H) and answers the depth question directly:

  Where in the network does step-correctness structure live, and what does the 2D
  map look like as you climb the layers?

For each requested (layer, token) plane it:
  - embeds all n steps on the FULL hidden dims with UMAP (cosine, no PCA bottleneck),
  - colours the map by binary step-correctness,
  - measures label-decodability (5-fold balanced logreg accuracy vs majority floor) —
    the quantitative "is correctness linearly readable here" signal.

Outputs (results/prm800k_layers/):
  - dashboard.html                combined page: UMAP scatter "in space" (left) + the
                                  decodability-vs-depth graph (right), one dropdown
                                  flips the plane in both and moves the marker on the
                                  curve. Self-contained, open in a browser.
  - projection_layers.html        UMAP scatter only, dropdown flips between planes.
  - layer_decodability.csv        per-plane: layer, frac, token, decodability, floor,
                                  n_clusters.
  - layer_decodability.png        decodability vs layer depth, one line per token.

Usage:
    python scripts/analysis/s3_prm800k_layer_projection.py
    python scripts/analysis/s3_prm800k_layer_projection.py --tokens last
    python scripts/analysis/s3_prm800k_layer_projection.py --layers 11 20 28
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import HDBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, normalize
from umap import UMAP

from src.data.prm800k_val_data import load_prm800k_multitoken

DEFAULT_MULTITOKEN_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b/prm_multitoken")
ROOT = Path("results/prm800k_layers")
LABEL_COLOR = {0: "#3cb44b", 1: "#e6194B"}        # 0 = correct, 1 = incorrect
TOKEN_DASH = {"last": "-", "first": "--"}
TOKEN_MARK = {"last": "o", "first": "s"}


def classifiability(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """5-fold balanced logreg accuracy of predicting y from X, and majority floor."""
    _, counts = np.unique(y, return_counts=True)
    floor = counts.max() / len(y)
    if len(set(y.tolist())) < 2:
        return float("nan"), floor
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    acc = float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean())
    return acc, floor


def embed_plane(H: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    """Standardize -> unit-norm (cosine) -> UMAP 2D; also return the standardized X."""
    Xs = StandardScaler().fit_transform(H).astype(np.float32)
    Xn = normalize(Xs)
    emb = UMAP(n_components=2, random_state=args.seed, metric="cosine",
               n_neighbors=args.n_neighbors, min_dist=args.min_dist).fit_transform(Xn)
    return emb, Xs


def _scatter_traces(fig, p, *, size=4, row=None, col=None):
    """Add the correct/incorrect scatter pair for one plane; return their trace idxs."""
    idxs = []
    for c in (0, 1):
        m = p["label"] == c
        if not m.any():
            continue
        fig.add_trace(go.Scattergl(
            x=p["emb"][m, 0], y=p["emb"][m, 1], mode="markers",
            name="correct" if c == 0 else "incorrect",
            marker=dict(size=size, color=LABEL_COLOR[c], opacity=0.6),
            text=[p["hover"][j] for j in np.where(m)[0]], hoverinfo="text",
            showlegend=False, visible=False), row=row, col=col)
        idxs.append(len(fig.data) - 1)
    return idxs


def build_projection_html(planes, out_dir, stem) -> Path:
    """UMAP-only page: dropdown flips the (layer, token) plane (matches projection_all)."""
    fig = go.Figure()
    plane_traces = []
    for p in planes:
        idxs = _scatter_traces(fig, p)
        btn = (f"L{p['layer']} ({p['frac']:.2f}) / {p['token']}  "
               f"[decod {p['decod']:.2f} vs {p['floor']:.2f}]")
        plane_traces.append((btn, idxs))
    for ti in plane_traces[0][1]:
        fig.data[ti].visible = True
    buttons = []
    for btn, idxs in plane_traces:
        vis = [False] * len(fig.data)
        for ti in idxs:
            vis[ti] = True
        buttons.append(dict(label=btn, method="update",
                            args=[{"visible": vis},
                                  {"title": f"PRM800K {stem} - {btn}"}]))
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor="left",
                          y=1.12, yanchor="top")],
        title=f"PRM800K {stem} - {plane_traces[0][0]}",
        width=1150, height=820, template="plotly_white")
    out = out_dir / "projection_layers.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def build_dashboard_html(planes, tokens, out_dir, stem) -> Path:
    """Combined page: UMAP scatter (left) + decodability curve (right), one dropdown.

    The dropdown picks a (layer, token) plane: the left scatter swaps to that plane and
    a ring marker hops to the matching point on the right-hand curve, so the spatial map
    and the depth graph stay in sync.
    """
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.09,
        subplot_titles=("UMAP of step encodings (full dims, cosine)",
                        "step-correctness decodability vs depth"))

    floor = planes[0]["floor"]
    fracs = [p["frac"] for p in planes]

    # ---- right panel: persistent decodability curve (always visible) ----
    base_idx = []
    for tok in tokens:
        sub = sorted([p for p in planes if p["token"] == tok], key=lambda q: q["frac"])
        fig.add_trace(go.Scatter(
            x=[q["frac"] for q in sub], y=[q["decod"] for q in sub],
            mode="lines+markers", name=f"{tok} token",
            line=dict(dash="dash" if tok == "first" else "solid"),
            marker=dict(size=7),
            hovertemplate=f"{tok} token<br>depth %{{x:.2f}}<br>decod %{{y:.3f}}<extra></extra>"),
            row=1, col=2)
        base_idx.append(len(fig.data) - 1)
    fig.add_trace(go.Scatter(
        x=[min(fracs), max(fracs)], y=[floor, floor], mode="lines",
        name=f"floor {floor:.2f}", line=dict(color="#888", dash="dot"),
        hoverinfo="skip"), row=1, col=2)
    base_idx.append(len(fig.data) - 1)
    fig.update_xaxes(title_text="layer depth (fraction of network)", row=1, col=2)
    fig.update_yaxes(title_text="5-fold logreg decodability", row=1, col=2)

    # ---- left panel: per-plane scatter + matching ring on the curve -----
    plane_traces = []
    for p in planes:
        idxs = _scatter_traces(fig, p, size=4, row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[p["frac"]], y=[p["decod"]], mode="markers", showlegend=False,
            marker=dict(size=16, color="rgba(0,0,0,0)", symbol="circle",
                        line=dict(color="#000", width=2)),
            hoverinfo="skip", visible=False), row=1, col=2)
        idxs.append(len(fig.data) - 1)
        plane_traces.append((p, idxs))

    for ti in plane_traces[0][1]:
        fig.data[ti].visible = True

    n_traces = len(fig.data)

    def title_for(p):
        return (f"PRM800K {stem}  -  L{p['layer']} (depth {p['frac']:.2f}) / "
                f"{p['token']} token   ·   decodability {p['decod']:.3f} "
                f"vs floor {p['floor']:.2f}")

    buttons = []
    for p, idxs in plane_traces:
        vis = [False] * n_traces
        for bi in base_idx:
            vis[bi] = True
        for ti in idxs:
            vis[ti] = True
        buttons.append(dict(
            label=f"L{p['layer']} ({p['frac']:.2f}) / {p['token']}",
            method="update", args=[{"visible": vis}, {"title.text": title_for(p)}]))

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor="left",
                          y=1.16, yanchor="top", direction="down", showactive=True)],
        title=dict(text=title_for(planes[0]), x=0.5, xanchor="center", y=0.98),
        width=1400, height=760, template="plotly_white", margin=dict(t=130),
        legend=dict(orientation="h", x=1.0, xanchor="right", y=1.12))
    fig.add_annotation(
        text="green = correct · red = incorrect", showarrow=False,
        xref="x domain", yref="y domain", x=0.0, y=1.04, xanchor="left",
        font=dict(size=11, color="#555"), row=1, col=1)
    out = out_dir / "dashboard.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", type=Path, default=DEFAULT_MULTITOKEN_DIR,
                    help="dir holding {stem}_h.npy (4D), _y.npy, _meta.jsonl, _manifest.json")
    ap.add_argument("--stem", type=str, default="prm800k_heldout_test")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layer indices to sweep (default: all in manifest)")
    ap.add_argument("--tokens", type=str, nargs="*", default=None,
                    help="token planes to sweep, e.g. last first (default: all in manifest)")
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    ap.add_argument("--min_cluster_size", type=int, default=30)
    ap.add_argument("--n_neighbors", type=int, default=20)
    ap.add_argument("--min_dist", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.merged_dir / f"{args.stem}_manifest.json").read_text())
    all_layers = list(manifest["layer_indices"])
    frac_of = dict(zip(all_layers, manifest["layer_fracs"]))
    all_tokens = list(manifest["token_order"])
    layers = args.layers if args.layers else all_layers
    tokens = args.tokens if args.tokens else all_tokens
    for lyr in layers:
        if lyr not in all_layers:
            ap.error(f"layer {lyr} not in manifest layers {all_layers}")
    for tok in tokens:
        if tok not in all_tokens:
            ap.error(f"token {tok!r} not in manifest tokens {all_tokens}")
    print(f"[plan] {args.stem}: layers={layers} x tokens={tokens} "
          f"= {len(layers) * len(tokens)} planes")

    # ---- embed every plane once, keep what both the page and graph need --
    planes = []
    for tok in tokens:
        for lyr in layers:
            d = load_prm800k_multitoken(args.merged_dir, args.stem, lyr, tok)
            n = len(d)
            label = d.label
            print(f"\n[plane] L{lyr} ({frac_of[lyr]:.2f}) / {tok}  n={n} "
                  f"correct={int((label==0).sum())} incorrect={int((label==1).sum())}")
            emb, Xs = embed_plane(d.hidden, args)
            cl = HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(emb)
            n_clusters = len(set(cl.tolist())) - (1 if -1 in cl else 0)
            acc, floor = classifiability(Xs, label)
            print(f"  decodability={acc:.3f} (floor {floor:.3f})  clusters={n_clusters}")
            hover = [
                f"{d.uid[i]} | prob={d.problem_id[i]} step={d.step_idx[i]}"
                f"<br>label={'correct' if label[i]==0 else 'incorrect'} "
                f"rating={d.rating[i]:+d}"
                for i in range(n)
            ]
            planes.append(dict(
                layer=lyr, token=tok, frac=frac_of[lyr], n=n, emb=emb, label=label,
                hover=hover, decod=acc, floor=floor, n_clusters=n_clusters))

    rows = [{
        "layer": p["layer"], "layer_frac": round(p["frac"], 4), "token": p["token"],
        "n": p["n"], "decodability": round(p["decod"], 4), "floor": round(p["floor"], 4),
        "lift": round(p["decod"] - p["floor"], 4), "n_clusters": p["n_clusters"],
    } for p in planes]

    dash = build_dashboard_html(planes, tokens, args.out_dir, args.stem)
    html = build_projection_html(planes, args.out_dir, args.stem)

    with (args.out_dir / "layer_decodability.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- static decodability vs depth curve (for decks / quick look) ----
    plt.figure(figsize=(7, 4.5))
    for tok in tokens:
        sub = sorted([r for r in rows if r["token"] == tok], key=lambda r: r["layer_frac"])
        plt.plot([r["layer_frac"] for r in sub], [r["decodability"] for r in sub],
                 TOKEN_DASH.get(tok, "-"), marker=TOKEN_MARK.get(tok, "o"),
                 label=f"{tok} token")
    floor = rows[0]["floor"]
    plt.axhline(floor, color="#888", ls=":", lw=1, label=f"majority floor {floor:.2f}")
    plt.xlabel("layer depth (fraction of network)")
    plt.ylabel("5-fold logreg decodability")
    plt.title(f"Step-correctness decodability vs depth\nPRM800K {args.stem} (7B, full dims)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    png = args.out_dir / "layer_decodability.png"
    plt.savefig(png, dpi=130)
    plt.close()

    print(f"\n[done] wrote -> {dash}   <-- open this")
    print(f"        + {html}")
    print(f"        + {args.out_dir / 'layer_decodability.csv'}")
    print(f"        + {png}")


if __name__ == "__main__":
    main()
