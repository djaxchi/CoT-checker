#!/usr/bin/env python3
"""Figures for context_ablation_v3 from the eval outputs.

Reads <results_dir>/{v1human,v2ds}/{leaderboard.csv,paired_differences.json}
and writes <results_dir>/figures/*.png:
  headline.png    seed-mean F1 / F1_PB per context, in-domain, other dataset, PB
  pre_post.png    rp_test AUROC on pre-error vs post-error steps per context
  contrasts.png   paired contrasts (seed-mean, 95% CI) on PB F1_PB and step F1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CTX = ("full", "prev1", "q", "none")
COLORS = {"full": "#1f4e79", "prev1": "#4f8fc0", "q": "#e08a2c", "none": "#b0b0b0"}
SOURCES = {"v1human": "train: PRM800K human", "v2ds": "train: ReProbe DeepSeek"}
# (column, label) per source: in-domain step F1, other-dataset step F1, PB mean F1_PB
PANELS = {
    "v1human": [("prm_test_f1", "in-domain F1\n(PRM800K human test)"),
                ("rp_test_f1", "other-dataset F1\n(ReProbe test)"),
                ("pb_mean_F1_PB", "ProcessBench\nmean F1_PB")],
    "v2ds": [("prm_test_f1", "in-domain F1\n(ReProbe test)"),
             ("prm_human_test_f1", "other-dataset F1\n(PRM800K human test)"),
             ("pb_mean_F1_PB", "ProcessBench\nmean F1_PB")],
}
PB = ("pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath")
CONTRASTS = ("q-full", "none-full", "prev1-full", "q-prev1")


def headline(res: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (src, title) in zip(axes, SOURCES.items()):
        d = pd.read_csv(res / src / "leaderboard.csv")
        g = d.groupby("arm")
        x = np.arange(len(PANELS[src]))
        w = 0.2
        for i, c in enumerate(CTX):
            m = [g[col].mean()[c] for col, _ in PANELS[src]]
            lo = [m[j] - g[col].min()[c] for j, (col, _) in enumerate(PANELS[src])]
            hi = [g[col].max()[c] - m[j] for j, (col, _) in enumerate(PANELS[src])]
            ax.bar(x + (i - 1.5) * w, m, w, yerr=[lo, hi], color=COLORS[c], label=c, capsize=2)
        base = d["prm_test_always_pos_f1"].iloc[0]
        ax.hlines(base, -0.45, 0.45, colors="k", linestyles=":", label="always-positive (in-domain)")
        ax.set_xticks(x, [lab for _, lab in PANELS[src]])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("seed mean (bars: seed min/max)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=5)
    fig.suptitle("Backbone context at encode time: less context lowers every metric")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "headline.png", dpi=150)
    plt.close(fig)


def pre_post(res: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (src, title) in zip(axes, SOURCES.items()):
        g = pd.read_csv(res / src / "leaderboard.csv").groupby("arm")
        x = np.arange(2)
        for i, c in enumerate(CTX):
            m = [g["rp_test_pre_auroc"].mean()[c], g["rp_test_post_auroc"].mean()[c]]
            ax.bar(x + (i - 1.5) * 0.2, m, 0.2, color=COLORS[c], label=c)
        ax.set_xticks(x, ["steps up to first error", "steps after first error"])
        ax.set_ylim(0.5, 1.0)
        ax.set_title(f"{title}\nReProbe test AUROC")
        ax.grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pre_post.png", dpi=150)
    plt.close(fig)


def contrasts(res: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, (src, title) in zip(axes, SOURCES.items()):
        p = json.loads((res / src / "paired_differences.json").read_text())
        rows = []
        for split, d in p.items():
            metric = "F1_PB" if split in PB else "f1"
            for c in CONTRASTS:
                v = d.get(c, {}).get(metric)
                if v:
                    rows.append((f"{split} {metric}", c, v["mean"], *v["mean_ci95"]))
        labels = list(dict.fromkeys(r[0] for r in rows))
        for k, c in enumerate(CONTRASTS):
            rr = [r for r in rows if r[1] == c]
            y = [labels.index(r[0]) + (k - 1.5) * 0.18 for r in rr]
            ax.errorbar([r[2] for r in rr], y,
                        xerr=[[r[2] - r[3] for r in rr], [r[4] - r[2] for r in rr]],
                        fmt="o", ms=4, capsize=2, label=c)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("paired difference (seed mean, 95% CI)")
        ax.grid(axis="x", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "contrasts.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=Path("results/context_ablation_v3"))
    a = ap.parse_args()
    out = a.results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    headline(a.results_dir, out)
    pre_post(a.results_dir, out)
    contrasts(a.results_dir, out)
    for f in sorted(out.glob("*.png")):
        print(f)


if __name__ == "__main__":
    main()
