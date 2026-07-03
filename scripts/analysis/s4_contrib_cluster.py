"""S4 contrib-cluster stage 5: cluster every representation and interpret.

For each (representation, layer): L2-normalize -> PCA-50 -> HDBSCAN (fallback:
KMeans over k in {10,20,40}, best silhouette). Regex tags are used ONLY after
clustering, for interpretation. UMAP is visualization only.

Outputs (under --run_dir):
  clusters/clusters_<repr>_layer_<L>.parquet        row_id -> cluster_id
  clusters/cluster_summary_<repr>_layer_<L>.csv     per-cluster stats
  clusters/tag_enrichment_<repr>_layer_<L>.csv      P(t|c)/P(t) per cluster x tag
  clusters/metrics_<repr>_layer_<L>.json            headline quality metrics
  cluster_cards_<repr>_layer_<L>.md                 manual inspection cards
  plots/umap_<repr>_layer_<L>_{clusters,tags,step_index,length}.png

Usage:
  python scripts/analysis/s4_contrib_cluster.py --run_dir runs/contrib_cluster \
    --reprs state qres contribution --layers 20 28
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import davies_bouldin_score, silhouette_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.contrib_cluster import (  # noqa: E402
    REPR_NAMES,
    TAG_NAMES,
    l2_normalize,
    surface_eta_squared,
    tag_enrichment,
    tag_entropy,
)

SURFACE_COLS = ["char_len", "token_count", "n_digits", "n_equals", "n_math_ops",
                "step_index", "relative_step_index"]


def cluster_vectors(Xp: np.ndarray, min_cluster_size: int, seed: int) -> tuple[np.ndarray, str]:
    """HDBSCAN on PCA vectors; fallback to best-silhouette KMeans."""
    try:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(Xp)
        return labels, "hdbscan"
    except ImportError:
        pass
    from sklearn.cluster import KMeans
    best, best_sil = None, -2.0
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Xp), size=min(10000, len(Xp)), replace=False)
    for k in (10, 20, 40):
        lab = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Xp)
        sil = silhouette_score(Xp[sub], lab[sub])
        if sil > best_sil:
            best, best_sil = lab, sil
    return best, "kmeans"


def quality_metrics(Xp: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    ids = sorted(c for c in set(labels.tolist()) if c >= 0)
    m = labels >= 0
    out = {
        "n_clusters": len(ids),
        "n_points": int(len(labels)),
        "n_noise": int((~m).sum()),
        "noise_ratio": float((~m).mean()),
        "cluster_sizes": {int(c): int((labels == c).sum()) for c in ids},
        "silhouette": None,
        "davies_bouldin": None,
    }
    if len(ids) >= 2 and m.sum() > len(ids):
        rng = np.random.default_rng(seed)
        idx = np.where(m)[0]
        if len(idx) > 10000:
            idx = rng.choice(idx, size=10000, replace=False)
        out["silhouette"] = float(silhouette_score(Xp[idx], labels[idx]))
        out["davies_bouldin"] = float(davies_bouldin_score(Xp[m], labels[m]))
    return out


def truncate(text: str, n: int = 220) -> str:
    text = " ".join(text.split())
    return text[: n - 1] + "…" if len(text) > n else text


def cluster_cards(labels, Xp, tags_df, enr_rows, ents, out_path: Path,
                  title: str, max_cards: int, min_card_size: int, seed: int) -> None:
    corpus_surface = {c: float(tags_df[c].mean()) for c in SURFACE_COLS}
    enr_by_cluster: dict[int, list[dict]] = {}
    for r in enr_rows:
        enr_by_cluster.setdefault(r["cluster"], []).append(r)
    ids = sorted((c for c in set(labels.tolist()) if c >= 0),
                 key=lambda c: -(labels == c).sum())
    rng = random.Random(seed)
    lines = [f"# Cluster cards: {title}", ""]
    for c in ids[:max_cards]:
        m = labels == c
        size = int(m.sum())
        if size < min_card_size:
            continue
        idxs = np.where(m)[0]
        centroid = Xp[m].mean(axis=0)
        d = np.linalg.norm(Xp[m] - centroid, axis=1)
        nearest = idxs[np.argsort(d)[:10]]
        rand = rng.sample(idxs.tolist(), min(10, size))
        top_enr = sorted(
            (r for r in enr_by_cluster.get(int(c), [])
             if r["p_tag_given_cluster"] >= 0.05),
            key=lambda r: -(r["enrichment"] if r["enrichment"] == r["enrichment"] else 0))[:5]
        lines += [
            f"## Cluster {c}  (n={size}, {size / len(labels):.1%} of steps, "
            f"tag-entropy={ents.get(int(c), float('nan')):.2f} bits)",
            "",
            "**Top enriched tags** (enrichment = P(t|c)/P(t)):",
            "",
        ]
        for r in top_enr:
            lines.append(f"- {r['tag']}: x{r['enrichment']:.2f} "
                         f"(P(t|c)={r['p_tag_given_cluster']:.2f}, P(t)={r['p_tag']:.2f})")
        if not top_enr:
            lines.append("- (no tag reaches 5% within-cluster frequency)")
        lines += ["", "**Surface stats (cluster mean vs corpus mean):**", ""]
        for col in SURFACE_COLS:
            cm = float(tags_df.loc[m, col].mean())
            lines.append(f"- {col}: {cm:.2f} vs {corpus_surface[col]:.2f}")
        lines += ["", "**10 nearest-to-centroid steps:**", ""]
        lines += [f"1. {truncate(t)}" for t in tags_df.loc[nearest, 'step_text']]
        lines += ["", "**10 random steps:**", ""]
        lines += [f"1. {truncate(t)}" for t in tags_df.loc[rand, 'step_text']]
        lines.append("")
    out_path.write_text("\n".join(lines))


def umap_plots(Xp, labels, tags_df, plots_dir: Path, stem: str, seed: int) -> list[Path]:
    from umap import UMAP
    emb = UMAP(n_components=2, random_state=seed, metric="cosine").fit_transform(Xp)
    written = []

    def scatter(color_vals, fname, title, categorical, legend_names=None, log=False):
        fig, ax = plt.subplots(figsize=(9, 7.5))
        if categorical:
            cats = legend_names or sorted(set(color_vals.tolist()))
            cmap = plt.get_cmap("tab20")
            for i, cat in enumerate(cats):
                m = color_vals == cat
                if not m.any():
                    continue
                is_noise = str(cat) in ("-1", "NONE")
                ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5,
                           color="#b0b0b0" if is_noise else cmap(i % 20),
                           label=f"{cat} ({int(m.sum())})", rasterized=True)
            ax.legend(markerscale=4, fontsize=6, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), frameon=False)
        else:
            v = np.log10(np.maximum(color_vals.astype(float), 1)) if log else color_vals
            sc = ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.5, c=v,
                            cmap="viridis", rasterized=True)
            fig.colorbar(sc, ax=ax, shrink=0.8,
                         label=("log10 " if log else "") + title.split(" by ")[-1])
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout()
        p = plots_dir / fname
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    scatter(labels.astype(str), f"umap_{stem}_clusters.png",
            f"{stem} by cluster",
            categorical=True,
            legend_names=[str(c) for c in
                          sorted(set(labels.tolist()), key=lambda c: (c < 0, c))])
    scatter(tags_df["top_tag"].to_numpy(), f"umap_{stem}_tags.png",
            f"{stem} by top regex tag", categorical=True,
            legend_names=list(TAG_NAMES) + ["NONE"])
    scatter(tags_df["step_index"].to_numpy(), f"umap_{stem}_step_index.png",
            f"{stem} by step_index", categorical=False)
    scatter(tags_df["char_len"].to_numpy(), f"umap_{stem}_length.png",
            f"{stem} by step length (chars)", categorical=False, log=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/contrib_cluster"))
    ap.add_argument("--reprs", type=str, nargs="+", default=list(REPR_NAMES))
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 28])
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--min_cluster_size", type=int, default=50)
    ap.add_argument("--max_cards", type=int, default=15)
    ap.add_argument("--min_card_size", type=int, default=50)
    ap.add_argument("--no_umap", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reprs_dir = args.run_dir / "reprs"
    clusters_dir = args.run_dir / "clusters"
    plots_dir = args.run_dir / "plots"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(reprs_dir / "step_metadata.parquet")
    tags = pd.read_parquet(args.run_dir / "tags.parquet")
    assert (tags["row_id"].to_numpy() == meta["row_id"].to_numpy()).all(), \
        "tags.parquet and step_metadata.parquet are misaligned"
    tags_df = tags.copy()
    tags_df["step_text"] = meta["step_text"].to_numpy()
    tag_matrix = tags_df[[f"tag_{n}" for n in TAG_NAMES]].to_numpy(dtype=bool)

    for name in args.reprs:
        for li in args.layers:
            stem = f"{name}_layer_{li}"
            norm_path = reprs_dir / f"repr_{name}_norm_layer_{li}.npy"
            if norm_path.exists():
                X = np.load(norm_path).astype(np.float32)
            else:
                X = l2_normalize(np.load(reprs_dir / f"repr_{name}_layer_{li}.npy")
                                 .astype(np.float32))
            print(f"[cluster] {stem}: {X.shape} -> PCA{args.pca}", flush=True)
            Xp = PCA(n_components=min(args.pca, X.shape[1]),
                     random_state=args.seed).fit_transform(X)
            labels, algo = cluster_vectors(Xp, args.min_cluster_size, args.seed)

            metrics = quality_metrics(Xp, labels, args.seed)
            metrics.update({
                "repr": name, "layer": li, "algo": algo,
                "created": datetime.now(timezone.utc).isoformat(),
            })
            enr_rows = tag_enrichment(labels, tag_matrix)
            ents = tag_entropy(labels, tags_df["top_tag"].tolist())
            metrics["surface_eta2"] = {
                col: surface_eta_squared(labels, tags_df[col].to_numpy())
                for col in SURFACE_COLS
            }
            # size-weighted mean tag entropy and mean max-enrichment over clusters
            ids = [c for c in metrics["cluster_sizes"]]
            if ids:
                sizes = np.array([metrics["cluster_sizes"][c] for c in ids], float)
                w = sizes / sizes.sum()
                metrics["weighted_tag_entropy"] = float(sum(
                    wi * ents[int(c)] for wi, c in zip(w, ids)))
                max_enr = []
                for c in ids:
                    vals = [r["enrichment"] for r in enr_rows
                            if r["cluster"] == int(c)
                            and r["p_tag_given_cluster"] >= 0.05
                            and r["enrichment"] == r["enrichment"]]
                    max_enr.append(max(vals) if vals else 0.0)
                metrics["weighted_max_enrichment"] = float(np.dot(w, max_enr))
            else:
                metrics["weighted_tag_entropy"] = None
                metrics["weighted_max_enrichment"] = None

            pd.DataFrame({"row_id": meta["row_id"], "cluster": labels}).to_parquet(
                clusters_dir / f"clusters_{stem}.parquet", index=False)
            pd.DataFrame(enr_rows).to_csv(
                clusters_dir / f"tag_enrichment_{stem}.csv", index=False)

            summary_rows = []
            for c in sorted(set(labels.tolist())):
                m = labels == c
                row = {
                    "repr": name, "layer": li, "cluster": int(c),
                    "n": int(m.sum()), "share": float(m.mean()),
                    "tag_entropy_bits": ents[int(c)],
                }
                for col in SURFACE_COLS:
                    row[f"mean_{col}"] = float(tags_df.loc[m, col].mean())
                top = sorted((r for r in enr_rows if r["cluster"] == int(c)
                              and r["p_tag_given_cluster"] >= 0.05),
                             key=lambda r: -(r["enrichment"]
                                             if r["enrichment"] == r["enrichment"] else 0))[:3]
                for k, r in enumerate(top, 1):
                    row[f"top_tag_{k}"] = r["tag"]
                    row[f"top_tag_{k}_enrichment"] = round(r["enrichment"], 2)
                summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(
                clusters_dir / f"cluster_summary_{stem}.csv", index=False)
            (clusters_dir / f"metrics_{stem}.json").write_text(
                json.dumps(metrics, indent=2))

            cluster_cards(labels, Xp, tags_df, enr_rows, ents,
                          args.run_dir / f"cluster_cards_{stem}.md",
                          stem, args.max_cards, args.min_card_size, args.seed)

            if not args.no_umap:
                try:
                    written = umap_plots(Xp, labels, tags_df, plots_dir, stem,
                                         args.seed)
                    print(f"[cluster] {stem}: {len(written)} plots", flush=True)
                except ImportError as e:
                    print(f"[cluster] {stem}: umap unavailable ({e}); "
                          "skipping plots", flush=True)
            print(f"[cluster] {stem}: algo={algo} clusters={metrics['n_clusters']} "
                  f"noise={metrics['noise_ratio']:.1%} "
                  f"sil={metrics['silhouette']} "
                  f"wme={metrics['weighted_max_enrichment']}", flush=True)

    print(f"[cluster] done -> {clusters_dir}/, {plots_dir}/", flush=True)


if __name__ == "__main__":
    main()
