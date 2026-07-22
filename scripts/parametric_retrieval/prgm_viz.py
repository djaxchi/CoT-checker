"""parametric_retrieval_minimal_v1 figure (local, no GPU).

Reads expF/{curves.csv, neuron_recurrence.csv, greedy_summary.json} and writes
one 3-panel PNG answering "what is really making the model flip":

  A  coordinate sparsity (residual, full layer): recovery vs number of
     injected coordinates, top-|Delta| vs random vs shared-subspace rank.
     Sparse specific coordinates or a distributed / low-rank direction?
  B  MLP neurons (mlp layer): recovery vs number of injected neurons, ranked
     by gradient attribution vs magnitude vs random, with the median greedy
     minimal-set size marked. How few neurons carry the flip?
  C  neuron recurrence across facts: fraction of pairs each neuron appears in
     the top attribution set (sorted). Concentrated = a shared mechanism;
     flat = fact-specific memories.

  python scripts/parametric_retrieval/prgm_viz.py \
      --run_dir runs/parametric_retrieval_access_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RANK_COLOR = {"topk_mag": "#1f77b4", "random": "#7f8c8d",
              "subspace": "#9467bd", "attr": "#d62728", "magnitude": "#2ca02c"}


def _line(ax, g, label, color):
    g = g.sort_values("k")
    ax.plot(g.k, g.recovery, "-o", ms=3, color=color, label=label)
    ax.fill_between(g.k, g.lo, g.hi, color=color, alpha=0.15)


def panel_coord(ax, curves):
    c = curves[curves.task == "coord"]
    for ranking in ["topk_mag", "random", "subspace"]:
        g = c[c.ranking == ranking]
        if not g.empty:
            lab = {"topk_mag": "top-|Δ| coords", "random": "random coords",
                   "subspace": "shared subspace (rank)"}[ranking]
            _line(ax, g, lab, RANK_COLOR[ranking])
    ax.set_xscale("log")
    ax.set_xlabel("# coordinates injected  (or subspace rank)")
    ax.set_ylabel("failures recovered (gold -> rank 1)")
    ax.set_title("A. Residual edit: sparse coords\nvs distributed direction",
                 fontsize=10)
    ax.legend(fontsize=8)


def panel_neuron(ax, curves, greedy):
    c = curves[curves.task == "neuron"]
    for ranking in ["attr", "magnitude", "random"]:
        g = c[c.ranking == ranking]
        if not g.empty:
            lab = {"attr": "gradient attribution", "magnitude": "|Δg|·‖W‖",
                   "random": "random neurons"}[ranking]
            _line(ax, g, lab, RANK_COLOR[ranking])
    if greedy and greedy.get("median_min_k"):
        ax.axvline(greedy["median_min_k"], color="k", ls="--", lw=1.2,
                   label=f"greedy median = {greedy['median_min_k']:.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("# MLP neurons injected")
    ax.set_ylabel("failures recovered (gold -> rank 1)")
    ax.set_title("B. MLP neurons: how few\ncarry the flip", fontsize=10)
    ax.legend(fontsize=8)


def panel_recurrence(ax, rec):
    if rec is None or rec.empty:
        ax.text(0.5, 0.5, "no recurrence data", ha="center")
        return
    y = rec.frac_pairs.to_numpy()
    ax.plot(np.arange(1, len(y) + 1), y, color="#d62728")
    ax.fill_between(np.arange(1, len(y) + 1), 0, y, color="#d62728", alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("neuron rank (by pairs it appears in)")
    ax.set_ylabel("fraction of pairs with neuron in top set")
    ax.set_title("C. Neuron recurrence across facts:\nshared vs fact-specific",
                 fontsize=10)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--out", type=Path, default=Path(
        "results/parametric_retrieval_minimal_v1/prgm_v1_results.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    exp = args.run_dir / "expF"

    curves = pd.read_csv(exp / "curves.csv")
    rec = pd.read_csv(exp / "neuron_recurrence.csv") \
        if (exp / "neuron_recurrence.csv").exists() else None
    greedy = json.loads((exp / "greedy_summary.json").read_text()) \
        if (exp / "greedy_summary.json").exists() else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panel_coord(axes[0], curves)
    panel_neuron(axes[1], curves, greedy)
    panel_recurrence(axes[2], rec)
    fig.suptitle("parametric_retrieval_minimal_v1: what minimal subset flips "
                 "the answer (Qwen2.5-7B-Instruct)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=150)
    print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    main()
