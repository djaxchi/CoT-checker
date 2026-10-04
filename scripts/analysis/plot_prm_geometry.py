"""Figures for prm_geometry_v1 (outputs of scripts/analysis/prm_geometry.py).

    python scripts/analysis/plot_prm_geometry.py --geo_dir <geometry> --out_dir results/prm_geometry_v1

Writes:
  methods_auroc.png      test AUROC of every closed-form scorer, PRM vs Instruct, per readout
  lda_topk.png           LDA restricted to the top-K PCs: how many dimensions separation needs
  pc_auroc.png           per-PC test AUROC (unsupervised components, variance order)
  views_<rep>.png        test steps in 2-D (top-2 PCs; LDA axis x top orthogonal PC) and
                         the LDA score per class, for each backbone
  summary.md             the numbers behind the figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BACKBONE = {"prm": ("#2a78d6", "Qwen2.5-Math-PRM-7B"), "instruct": ("#eb6834", "Qwen3-8B Instruct")}
CLASS = {0: ("#2a78d6", "correct step"), 1: ("#eb6834", "incorrect step")}
METHODS = ["pc_best", "mean_diff", "centroid_cos", "knn", "cov_only", "qda", "lda"]
METHOD_LABEL = {"pc_best": "best single PC\n(unsupervised)", "mean_diff": "class-mean axis",
                "centroid_cos": "nearest centroid\n(cosine)", "knn": "kNN vote (k=50)",
                "cov_only": "covariance only\n(2nd order)", "qda": "QDA\n(Gaussian, 2 cov)",
                "lda": "LDA\n(whitened mean axis)"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load(geo: Path):
    out = {}
    for mf in sorted(geo.glob("*/*/metrics.json")):
        bb, rep = mf.parent.parent.name, mf.parent.name
        out[(bb, rep)] = (json.loads(mf.read_text()), np.load(mf.parent / "arrays.npz"))
    return out


def fig_methods(data, reps, backbones, path, learned=None):
    fig, axes = plt.subplots(1, len(reps), figsize=(4.6 * len(reps), 4.2), sharey=True,
                             facecolor=SURF)
    axes = np.atleast_1d(axes)
    for ax, rep in zip(axes, reps):
        style(ax)
        x = np.arange(len(METHODS))
        w = 0.36
        for i, bb in enumerate(backbones):
            if (bb, rep) not in data:
                continue
            m = data[(bb, rep)][0]["methods"]
            vals = [m[k]["auroc_test"] if k in m else np.nan for k in METHODS]
            ax.bar(x + (i - 0.5) * w, vals, w - 0.03, color=BACKBONE[bb][0],
                   label=BACKBONE[bb][1], edgecolor=SURF, linewidth=1)
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.text(len(METHODS) - 0.5, 0.505, "chance", color=INK2, fontsize=7, ha="right")
        for bb in backbones:
            best = (learned or {}).get((bb, rep))
            if best is not None:
                ax.axhline(best, color=BACKBONE[bb][0], linewidth=1.2, linestyle="--",
                           label=f"{BACKBONE[bb][1]}: best trained probe")
        ax.set_xticks(x, [METHOD_LABEL[k] for k in METHODS], rotation=55, ha="right", fontsize=7.5)
        ax.set_ylim(0.45, 1.0)
        ax.set_title(rep, color=INK, fontsize=10)
    axes[0].set_ylabel("PRM800K test AUROC (no trained learner)", color=INK, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=len(l))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_topk(data, reps, backbones, path):
    fig, axes = plt.subplots(1, len(reps), figsize=(4.4 * len(reps), 3.6), sharey=True,
                             facecolor=SURF)
    axes = np.atleast_1d(axes)
    for ax, rep in zip(axes, reps):
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for bb in backbones:
            if (bb, rep) not in data:
                continue
            kc = data[(bb, rep)][0]["lda_topk_pcs"]
            ks = sorted(int(k) for k in kc)
            ax.plot(ks, [kc[str(k)]["auroc_test"] for k in ks], color=BACKBONE[bb][0],
                    linewidth=2, marker="o", markersize=4, label=BACKBONE[bb][1])
        ax.set_xscale("log", base=2)
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xlabel("top-K principal components kept (log scale)", color=INK, fontsize=8.5)
        ax.set_title(rep, color=INK, fontsize=10)
        ax.set_ylim(0.45, 1.0)
    axes[0].set_ylabel("LDA test AUROC", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_pc(data, reps, backbones, path):
    fig, axes = plt.subplots(len(backbones), len(reps), figsize=(4.4 * len(reps), 2.6 * len(backbones)),
                             sharey=True, facecolor=SURF, squeeze=False)
    for r, bb in enumerate(backbones):
        for c, rep in enumerate(reps):
            ax = axes[r, c]
            style(ax)
            if (bb, rep) not in data:
                continue
            arr = data[(bb, rep)][1]
            au = np.abs(arr["pc_auroc_test"] - 0.5) + 0.5
            ax.bar(np.arange(1, len(au) + 1), au - 0.5, bottom=0.5, width=1.0,
                   color=BACKBONE[bb][0], linewidth=0)
            ax.set_ylim(0.5, 0.8)
            ax.set_xlim(0, len(au) + 1)
            if r == 0:
                ax.set_title(rep, color=INK, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{BACKBONE[bb][1]}\nmax(AUROC, 1-AUROC)", color=INK, fontsize=8)
            if r == len(backbones) - 1:
                ax.set_xlabel("principal component (variance rank)", color=INK, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_views(data, rep, backbones, path, seed=0):
    rows = [bb for bb in backbones if (bb, rep) in data]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.9 * len(rows)), facecolor=SURF,
                             squeeze=False)
    for r, bb in enumerate(rows):
        metrics, arr = data[(bb, rep)]
        y = arr["y_test"].astype(int)
        order = np.random.default_rng(seed).permutation(len(y))
        for c, (key, title, xl, yl) in enumerate([
                ("pca2_test", "top-2 principal components (unsupervised)", "PC1", "PC2"),
                ("lda2_test", "LDA axis x top orthogonal PC", "LDA axis", "PC1 (orthogonal to LDA)")]):
            ax = axes[r, c]
            style(ax)
            ax.grid(False)
            P = arr[key]
            ax.scatter(P[order, 0], P[order, 1], s=6, c=[CLASS[v][0] for v in y[order]],
                       alpha=0.55, linewidths=0)
            ax.set_xlabel(xl, color=INK, fontsize=8.5)
            ax.set_ylabel(yl, color=INK, fontsize=8.5)
            ax.set_title(f"{BACKBONE[bb][1]}: {title}", color=INK, fontsize=9)
        ax = axes[r, 2]
        style(ax)
        s = arr["score_test__lda"]
        bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 50)
        for v in (0, 1):
            ax.hist(s[y == v], bins=bins, color=CLASS[v][0], alpha=0.55, label=CLASS[v][1])
        au = metrics["methods"]["lda"]["auroc_test"]
        ax.set_title(f"{BACKBONE[bb][1]}: LDA score, test AUROC {au:.3f}", color=INK, fontsize=9)
        ax.set_xlabel("LDA projection", color=INK, fontsize=8.5)
        ax.set_ylabel("steps", color=INK, fontsize=8.5)
        if r == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def summary(data, reps, backbones, path):
    lines = ["# prm_geometry_v1: closed-form separation, no trained learner", "",
             "Fit on PRM800K train (class means / covariances only); val picks shrinkage "
             "and threshold; test_2k and ProcessBench untouched. PB F1 = four-subset mean.", ""]
    for rep in reps:
        lines += [f"## {rep}", "",
                  "| scorer | " + " | ".join(f"{b} AUROC | {b} ID F1 | {b} PB calib20 | {b} PB oracle"
                                             for b in backbones) + " |",
                  "|---|" + "---:|" * (4 * len(backbones))]
        for k in METHODS:
            cells = []
            for b in backbones:
                m = data.get((b, rep), ({"methods": {}},))[0]["methods"].get(k)
                cells += (["-"] * 4 if m is None else
                          [f"{m['auroc_test']:.4f}", f"{m['id_f1_incorrect_val_selected']:.4f}",
                           f"{m['pb_avg_F1_PB_calib20']:.4f}", f"{m['pb_avg_F1_PB_oracle']:.4f}"])
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
        for b in backbones:
            if (b, rep) not in data:
                continue
            m = data[(b, rep)][0]
            mh, kc = m["mahalanobis"], m["lda_topk_pcs"]
            lines.append(
                f"- {b}: dim {m['dim']}, Mahalanobis D {mh['D']:.3f} (Gaussian AUROC "
                f"{mh['gaussian_auroc']:.4f}); D^2 participation ratio {mh['participation_ratio']:.1f}, "
                f"{mh['share_in_bottom50pct_variance_dirs']:.1%} of D^2 in the lower-variance half "
                f"of directions; best single PC #{m['pc_best_index'] + 1}; LDA on top-K PCs "
                + ", ".join(f"K={k}: {kc[k]['auroc_test']:.3f}" for k in sorted(kc, key=int))
                + f"; class covariance trace ratio (incorrect/correct) "
                f"{m['class_cov']['trace_ratio_incorrect_over_correct']:.3f}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geo_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--leaderboard", type=Path, default=None,
                   help="results/prm_backbone_v1/leaderboard.json: draws the best trained "
                        "probe's test AUROC per (backbone, readout) as a reference line")
    a = p.parse_args()
    data = load(a.geo_dir)
    backbones = [b for b in ("prm", "instruct") if any(k[0] == b for k in data)]
    order = ["last_token", "step_mean", "boundary_stats"]
    reps = [r for r in order if any(k[1] == r for k in data)] + sorted(
        {k[1] for k in data} - set(order))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    learned = {}
    if a.leaderboard:
        for row in json.loads(a.leaderboard.read_text()):
            for bb in ("prm", "instruct"):
                if row.get(bb):
                    k = (bb, row["rep"])
                    learned[k] = max(learned.get(k, 0.0), row[bb]["auroc"][0])
    fig_methods(data, reps, backbones, a.out_dir / "methods_auroc.png", learned)
    fig_topk(data, reps, backbones, a.out_dir / "lda_topk.png")
    fig_pc(data, reps, backbones, a.out_dir / "pc_auroc.png")
    for rep in reps:
        fig_views(data, rep, backbones, a.out_dir / f"views_{rep}.png")
    summary(data, reps, backbones, a.out_dir / "summary.md")
    for f in sorted(a.out_dir.iterdir()):
        print(f)


if __name__ == "__main__":
    main()
