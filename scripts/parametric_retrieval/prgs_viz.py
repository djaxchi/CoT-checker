"""parametric_retrieval_steer_v1 figure (local, no GPU).

Reads expG/{steer_curves.csv, specificity_matrix.npy, specificity_labels.json,
summary.json} and writes a 3-panel PNG:

  A  Golden-Gate curve: rate the fact's object/subject appears on UNRELATED
     prompts vs clamp strength alpha, steering the fact's neuron (target) vs a
     random neuron, with the own-question positive control.
  B  specificity matrix: steer fact i's neuron, does object j appear? Diagonal
     dominance = fact-specific "knowledge neuron"; flat = polysemantic /
     contextual.
  C  coherence (unique-token ratio) vs alpha: how fast steering degrades output.

  python scripts/parametric_retrieval/prgs_viz.py \
      --run_dir runs/parametric_retrieval_access_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALPHAS = ["none", "0", "3", "6", "10"]


def _xnum(a):
    return {"none": -1, "0": 0, "3": 3, "6": 6, "10": 10}[a]


def panel_curve(ax, curve, summary):
    for arm, color in (("target", "#d62728"), ("random", "#7f8c8d")):
        g = curve[curve.arm == arm].copy()
        g["x"] = g.alpha.map(_xnum)
        g = g.sort_values("x")
        ax.plot(g.x, g.hit_own, "-o", color=color, label=f"{arm} neuron")
    own = curve.drop_duplicates("alpha").copy()
    own["x"] = own.alpha.map(_xnum)
    own = own.sort_values("x")
    ax.plot(own.x, own.own_question_hit, "--", color="#2ca02c",
            label="own question (pos. ctrl)")
    ax.set_xlabel("clamp strength alpha (x max act); -1 = no clamp")
    ax.set_ylabel("fact object/subject appears (unrelated prompts)")
    ax.set_title("A. Does steering the neuron summon\nthe fact off-context?",
                 fontsize=10)
    ax.legend(fontsize=8)


def panel_matrix(ax, M, summary):
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=max(M.max(),
                                                                    1e-6))
    ax.set_xlabel("object j detected")
    ax.set_ylabel("steer fact i's neuron")
    d = summary.get("diag_mention_rate", 0)
    o = summary.get("offdiag_mention_rate", 0)
    ax.set_title(f"B. Specificity matrix\ndiag {d:.2f} vs off-diag {o:.2f}",
                 fontsize=10)
    im.axes.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def panel_coherence(ax, curve):
    for arm, color in (("target", "#d62728"), ("random", "#7f8c8d")):
        g = curve[curve.arm == arm].copy()
        g["x"] = g.alpha.map(_xnum)
        g = g.sort_values("x")
        ax.plot(g.x, g.uniq_ratio, "-o", color=color, label=f"{arm} neuron")
    ax.set_xlabel("clamp strength alpha")
    ax.set_ylabel("unique-token ratio (coherence)")
    ax.set_title("C. Output coherence vs\nclamp strength", fontsize=10)
    ax.legend(fontsize=8)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--out", type=Path, default=Path(
        "results/parametric_retrieval_steer_v1/prgs_v1_results.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    exp = args.run_dir / "expG"

    curve = pd.read_csv(exp / "steer_curves.csv")
    M = np.load(exp / "specificity_matrix.npy")
    summary = json.loads((exp / "summary.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panel_curve(axes[0], curve, summary)
    panel_matrix(axes[1], M, summary)
    panel_coherence(axes[2], curve)
    fig.suptitle("parametric_retrieval_steer_v1: Golden-Gate-style neuron "
                 "steering of flip-neurons (Qwen2.5-7B-Instruct)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=150)
    print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    main()
