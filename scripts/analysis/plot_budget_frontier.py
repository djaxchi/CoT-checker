#!/usr/bin/env python3
"""Two panels: accuracy against the sample budget, and what the verifier adds.

The accuracy panel carries the three answer rules on one axis, since they are the
same measure. The gain is a second panel rather than a second y-scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
PLAIN, VERIFIER, CHEAP = "#2a78d6", "#eb6834", "#1baf7a"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path,
                   default=Path("results/onpolicy_budget/budget_frozen_ver_clean.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/onpolicy_budget/budget_frontier.png"))
    args = p.parse_args()

    rep = json.loads(args.results.read_text())
    by_n = {int(k): v for k, v in rep["by_n"].items()}
    ns = sorted(by_n)
    plain = [by_n[n]["majority"] for n in ns]
    ver = [by_n[n]["majority_tie"] for n in ns]
    cheap = [by_n[n]["majority_cheap"] for n in ns]
    gain = [by_n[n]["gain_vs_majority"]["delta"] for n in ns]
    lo = [g - by_n[n]["gain_vs_majority"]["ci95"][0] for g, n in zip(gain, ns)]
    hi = [by_n[n]["gain_vs_majority"]["ci95"][1] - g for g, n in zip(gain, ns)]

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(7.4, 6.6), sharex=True, height_ratios=[2.1, 1],
        gridspec_kw={"hspace": 0.14})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#d5d4cf")
        a.tick_params(colors=INK_2, labelsize=9)

    # Direct labels as well as a legend, nudged apart where the curves converge.
    for series, color, label, dy in (
            (ver, VERIFIER, "verifier breaks ties", 0),
            (plain, PLAIN, "ties by chance", 7),
            (cheap, CHEAP, "shortest breaks ties", -8)):
        ax.plot(ns, series, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=f"majority, {label}", zorder=3)
        ax.annotate(label, (ns[-1], series[-1]), xytext=(8, dy),
                    textcoords="offset points", color=color, fontsize=8.5,
                    va="center")
    ax.set_ylabel("final-answer accuracy", color=INK, fontsize=10)
    ax.set_title("A verifier is worth most when there are fewest samples",
                 color=INK, fontsize=12, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_2)
    ax.set_xlim(0.6, 13.4)

    bx.bar(ns, gain, color=VERIFIER, width=0.62, zorder=3)
    bx.errorbar(ns, gain, yerr=[lo, hi], fmt="none", ecolor=INK_2,
                elinewidth=1.2, capsize=3, zorder=4)
    bx.axhline(0, color="#d5d4cf", linewidth=1)
    bx.set_ylabel("gain over ties by chance", color=INK, fontsize=10)
    bx.set_xlabel("samples drawn per problem (N)", color=INK, fontsize=10)
    bx.set_xticks(ns)
    for n, g, h in zip(ns, gain, hi):
        if n in (2, 4, 10):
            bx.annotate(f"{g:+.3f}", (n, g + h), xytext=(0, 6), ha="center",
                        textcoords="offset points", color=INK_2, fontsize=8.5)

    sub = (f"{rep['n_problems']} PRM800K problems ({rep['n_questions']} question "
           f"texts), {len(rep['cells'])} scored cell(s), {rep['n_orders']} sampling "
           f"orders; 95% CI resampled by question")
    fig.text(0.012, 0.012, sub, color=INK_2, fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
