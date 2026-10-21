#!/usr/bin/env python3
"""Class separation of end-of-step states, PRM800K test and ProcessBench.

Reads scripts/analysis/prm_geometry.py outputs (rep last_token). The first three
rows are the comparison, one vector per step read at the end of the step:
  instruct  Qwen3-8B Instruct, last step token, hidden_states[35] (block 34 of 36)
  rt        Qwen3-8B with the trained read token <v> right after the step, <v>'s
            hidden_states[35]: same model and layer as `instruct`, one position later
  prm       Qwen2.5-Math-PRM-7B, last step token, hidden_states[27] (block 26 of 28)
and two more for context:
  rt_final  <v>'s final post-norm state (what its own head reads)
  drv       DRV's pooled 512-d embedding: a trained learner's output, not a step state

Every projection is fitted on PRM800K train only (class means and covariances,
shrinkage picked on val); test_2k and ProcessBench are projected, never fitted.
ProcessBench classes: the trace's first erroneous step against the steps before
it and the steps of correct traces; steps after the first error carry no label
and are left out.

    python scripts/analysis/plot_rep_separation.py --geo_dir <geometry> --out_dir results/read_token_v1/rep_separation

Writes:
  views_test.png   PRM800K test_2k: top-2 PCs, LDA axis x top orthogonal PC, LDA score
  views_pb.png     the same three panels on ProcessBench, four subsets pooled
  separation.png   test AUROC, ProcessBench AUROC and calib-20 F1_PB of the LDA score
  summary.md       the numbers behind the figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.plot_prm_geometry import INK, INK2, SURF, load, plt, style  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
REPS = {"instruct": ("#eb6834", "Qwen3-8B Instruct: last step token (L35)"),
        "rt": ("#3a9b5c", "Qwen3-8B + read token <v> (L35)"),
        "prm": ("#2a78d6", "Qwen2.5-Math-PRM-7B: last step token (L27)"),
        "rt_final": ("#7a9b3a", "read token <v>: final post-norm state"),
        "drv": ("#8a4fc2", "DRV: pooled embedding (learner output)")}
CLASS = {0: ("#2a78d6", "correct step"), 1: ("#eb6834", "incorrect step")}
PB_CLASS = {0: ("#2a78d6", "correct step (before the first error)"),
            1: ("#eb6834", "first erroneous step")}


def pb_pooled(arr):
    """(pca2, lda2, lda score, class) over the four subsets, labeled steps only."""
    cols = [np.concatenate([arr[f"{k}_pb_{s}" if k != "score" else f"score_pb_{s}__lda"]
                            for s in PB_SUBSETS]) for k in ("pca2", "lda2", "score", "y")]
    keep = cols[3] >= 0
    return [c[keep] for c in cols]


def pb_auroc(arr) -> dict[str, float]:
    out = {}
    for s in PB_SUBSETS:
        y, sc = arr[f"y_pb_{s}"], arr[f"score_pb_{s}__lda"]
        out[s] = float(roc_auc_score(y[y >= 0], sc[y >= 0]))
    out["mean"] = float(np.mean([out[s] for s in PB_SUBSETS]))
    return out


def lims(P, q=0.5):
    lo, hi = np.percentile(P, [q, 100 - q], axis=0)
    pad = 0.05 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def fig_views(data, names, path, pb: bool, seed: int = 0):
    fig, axes = plt.subplots(len(names), 3, figsize=(13.5, 3.7 * len(names)), facecolor=SURF,
                             squeeze=False)
    classes = PB_CLASS if pb else CLASS
    for r, bb in enumerate(names):
        metrics, arr = data[(bb, "last_token")]
        if pb:
            pca, lda2, score, y = pb_pooled(arr)
            au = pb_auroc(arr)["mean"]
            where = "ProcessBench (4 subsets)"
        else:
            pca, lda2, score, y = arr["pca2_test"], arr["lda2_test"], arr["score_test__lda"], arr["y_test"]
            au = metrics["methods"]["lda"]["auroc_test"]
            where = "PRM800K test"
        y = y.astype(int)
        # the rarer class is drawn last, so it is not buried
        order = np.random.default_rng(seed).permutation(len(y))
        order = np.concatenate([order[y[order] == 0], order[y[order] == 1]])
        for c, (P, title, xl, yl) in enumerate([
                (pca, "top-2 PCs (unsupervised)", "PC1", "PC2"),
                (lda2, "LDA axis x top orthogonal PC", "LDA axis (fit on PRM800K train)",
                 "PC1 orthogonal to LDA")]):
            ax = axes[r, c]
            style(ax)
            ax.grid(False)
            ax.scatter(P[order, 0], P[order, 1], s=4 if pb else 6,
                       c=[classes[v][0] for v in y[order]], alpha=0.35 if pb else 0.55, linewidths=0)
            xlim, ylim = lims(P)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(xl, color=INK, fontsize=8.5)
            ax.set_ylabel(yl, color=INK, fontsize=8.5)
            ax.set_title(f"{REPS[bb][1]}\n{title}", color=INK, fontsize=9)
        ax = axes[r, 2]
        style(ax)
        bins = np.linspace(np.percentile(score, 0.5), np.percentile(score, 99.5), 60)
        for v in (0, 1):
            ax.hist(score[y == v], bins=bins, color=classes[v][0], alpha=0.55, density=True,
                    label=f"{classes[v][1]} (n={int((y == v).sum())})")
        ax.set_title(f"{REPS[bb][1]}\nLDA score on {where}, AUROC {au:.3f}", color=INK, fontsize=9)
        ax.set_xlabel("LDA projection", color=INK, fontsize=8.5)
        ax.set_ylabel("density", color=INK, fontsize=8.5)
        if r == 0:
            ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)


def rows(data, names):
    out = []
    for bb in names:
        m, arr = data[(bb, "last_token")]
        lda = m["methods"]["lda"]
        out.append({"bb": bb, "dim": m["dim"], "test": lda["auroc_test"], "pb": pb_auroc(arr),
                    "calib20": lda["pb_avg_F1_PB_calib20"], "oracle": lda["pb_avg_F1_PB_oracle"],
                    "D": m["mahalanobis"]["D"], "pc_best": m["methods"]["pc_best"]["auroc_test"]})
    return out


def fig_bars(table, path):
    keys = [("pc_best", "best single PC\n(unsupervised), test AUROC"),
            ("test", "LDA, PRM800K\ntest AUROC"), ("pb", "LDA, ProcessBench\nAUROC (first error)"),
            ("calib20", "LDA, ProcessBench\ncalib-20 F1_PB")]
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURF)
    style(ax)
    x, w = np.arange(len(keys)), 0.8 / len(table)
    for i, r in enumerate(table):
        vals = [r[k]["mean"] if k == "pb" else r[k] for k, _ in keys]
        b = ax.bar(x + (i - (len(table) - 1) / 2) * w, vals, w - 0.02, color=REPS[r["bb"]][0],
                   label=REPS[r["bb"]][1], edgecolor=SURF)
        ax.bar_label(b, fmt="%.3f", fontsize=6.5, color=INK2, padding=1)
    ax.set_xticks(x, [k[1] for k in keys], fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("score (fit on PRM800K train only)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def summary(table, path):
    lines = ["# Class separation of step representations (closed form, fit on PRM800K train)", "",
             "LDA = shrunk within-class whitening of the class-mean axis, fit on PRM800K train, "
             "shrinkage on val. ProcessBench AUROC: first erroneous step vs the steps before it "
             "and the steps of correct traces (after-error steps left out).", "",
             "| representation | dim | best single PC test AUROC | LDA test AUROC | "
             "LDA PB AUROC (gsm8k / math / olymp / omni, mean) | LDA PB calib-20 F1_PB | "
             "LDA PB oracle F1_PB | Mahalanobis D |",
             "|---|---:|---:|---:|---|---:|---:|---:|"]
    for r in table:
        pb = " / ".join(f"{r['pb'][s]:.3f}" for s in PB_SUBSETS) + f", **{r['pb']['mean']:.3f}**"
        lines.append(f"| {REPS[r['bb']][1]} | {r['dim']} | {r['pc_best']:.3f} | {r['test']:.3f} | "
                     f"{pb} | {r['calib20']:.3f} | {r['oracle']:.3f} | {r['D']:.3f} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geo_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    a = p.parse_args()
    data = load(a.geo_dir)
    names = [b for b in REPS if (b, "last_token") in data]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig_views(data, names, a.out_dir / "views_test.png", pb=False)
    fig_views(data, names, a.out_dir / "views_pb.png", pb=True)
    table = rows(data, names)
    fig_bars(table, a.out_dir / "separation.png")
    summary(table, a.out_dir / "summary.md")
    for f in sorted(a.out_dir.iterdir()):
        print(f)


if __name__ == "__main__":
    main()
