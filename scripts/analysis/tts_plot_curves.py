#!/usr/bin/env python3
"""The sprint 7 budget curves: four answer rules against N and against tokens.

Recreated. The script that produced `figs/curves_n_tokens.png` for the sprint 7
deck is not in the tree or in git history, so the figure existed only as a PNG
and could not be restyled or corrected. This reproduces it from
`frontier.json`, which still holds the exact values the deck shows (gsm8k at
N=10: oracle 0.978, tie-break 0.929, majority 0.925, rerank 0.892, at 1224
generated tokens).

A missing rule is a hard error rather than a skipped line. A curve that
silently vanishes leaves a plot that still looks finished, and the whole point
of the figure is which rules sit above which.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROBE = "probe::step_tokens__transformer_d256_l2_f1024_h4__seed42"

# label, rule key, colour, linestyle, marker, linewidth, z-order
SERIES = [
    ("best possible choice", "oracle", "#A8A8A8", "--", None, 1.4, 1),
    ("majority vote", "majority", "#8A8A8A", "-", "o", 1.8, 2),
    ("verifier tie-break", f"tiebreak::{PROBE}::worst", "#C0392B", "-", "o", 2.0, 4),
    ("verifier rerank", f"rerank::{PROBE}::worst", "#2A6FB5", ":", "s", 1.8, 3),
]


def series_for(curves: list[dict], dataset: str, rule: str, ns: list[int]) -> list[float]:
    by_n = {r["n"]: r["accuracy"] for r in curves
            if r["dataset"] == dataset and r["split"] == "all" and r["rule"] == rule}
    missing = [n for n in ns if n not in by_n]
    if missing:
        raise SystemExit(
            f"{dataset}: rule {rule!r} has no value at N={missing}. Refusing to "
            f"draw a figure with a silently missing curve.")
    return [by_n[n] for n in ns]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frontier", type=Path,
                   default=Path("results/tts_roster_v1/frontier.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/tts_roster_v1/figs/curves_n_tokens_white.png"))
    p.add_argument("--datasets", nargs="+", default=["gsm8k", "math500"])
    p.add_argument("--facecolor", default="#ffffff",
                   help="figure and axes background; the deck version used an "
                        "off-white surface")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()

    curves = json.loads(a.frontier.read_text())["curves"]
    fig, axes = plt.subplots(1, len(a.datasets), figsize=(13.0, 5.4))
    axes = [axes] if len(a.datasets) == 1 else list(axes)
    fig.patch.set_facecolor(a.facecolor)

    for ax, ds in zip(axes, a.datasets):
        rows = [r for r in curves if r["dataset"] == ds and r["split"] == "all"]
        if not rows:
            raise SystemExit(f"no rows for dataset {ds!r} in {a.frontier}")
        ns = sorted({r["n"] for r in rows})
        ax.set_facecolor(a.facecolor)

        for label, rule, colour, style, marker, lw, z in SERIES:
            ax.plot(ns, series_for(curves, ds, rule, ns), style, color=colour,
                    marker=marker, markersize=5, linewidth=lw, label=label, zorder=z)

        ax.set_title(ds, fontsize=13, loc="left", color="#3a3a3a")
        ax.set_xlabel("N samples drawn", fontsize=12)
        ax.set_xticks(ns)
        ax.tick_params(labelsize=11, colors="#3a3a3a")
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#9a9a9a")

        # Tokens under N: the budget a practitioner actually spends, and the two
        # axes are not proportional across datasets.
        tok = {r["n"]: r["tokens_mean"] for r in rows if r["rule"] == "majority"}
        sec = ax.secondary_xaxis(-0.16)
        sec.set_xticks(ns)
        sec.set_xticklabels([f"{tok[n]:,.0f}".replace(",", " ") for n in ns],
                            fontsize=10, color="#6a6a6a")
        sec.set_xlabel("generated tokens per problem", fontsize=12, color="#3a3a3a")
        sec.spines["bottom"].set_visible(False)
        sec.tick_params(length=0, colors="#6a6a6a")

    axes[0].set_ylabel("final-answer accuracy", fontsize=12)
    axes[0].legend(frameon=False, fontsize=11, loc="lower right")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor=a.facecolor)
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
