#!/usr/bin/env python3
"""Figures of the read_token_v1 mechanistic study (docs/read_token_v1_plan.md, Stage G).

  m1   per-layer LDA (scripts/analysis/rth_layer_lda.py outputs m1_<rep>.json):
       ProcessBench calib-20 F1_PB and ProcessBench AUROC against relative depth for
       the Qwen3-8B last step token, the read token and the PRM's last step token.
       Row l (block l output) sits at depth (l + 1) / n_rows; the final post-norm
       state is the last row, at depth 1.
  m3   read-token attention mass per segment and layer (rth_mech_attention.py
       summary JSON), one panel per segment, one line per token.

    python scripts/analysis/rth_mech_plots.py m1 --mech results/read_token_v1/mech
    python scripts/analysis/rth_mech_plots.py m3 --mech results/read_token_v1/mech
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots 1-3 of the dataviz reference palette (light mode), fixed order
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
M1_REPS = (("last_step", "Qwen3-8B last step token"), ("read", "read token (RTH)"),
           ("prm", "PRM last step token"))


def depth_curve(m1: dict, key: str) -> tuple[list[float], list[float]]:
    n = m1["n_rows"]
    rows = sorted(int(k) for k in m1["layers"])
    return [(r + 1) / n for r in rows], [m1["layers"][str(r)][key] for r in rows]


def spread(ys: list[float], gap: float) -> list[float]:
    """Label heights for sorted values ys, at least `gap` apart."""
    out = []
    for y in ys:
        out.append(max(y, out[-1] + gap) if out else y)
    return out


def style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def plot_m1(mech: Path, out: Path) -> Path:
    m1 = {r: json.loads((mech / f"m1_{r}.json").read_text()) for r, _ in M1_REPS}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, (key, ylabel) in zip(axes, (("pb_calib20", "ProcessBench calib-20 F1_PB"),
                                        ("pb_auroc_mean", "ProcessBench AUROC"))):
        ends = []
        for (rep, label), color in zip(M1_REPS, SERIES):
            x, y = depth_curve(m1[rep], key)
            ax.plot(x, y, color=color, lw=2, label=label)
            ax.plot(x[-1], y[-1], "o", color=color, ms=8, mec="white", mew=2)
            ends.append((y[-1], x[-1]))
        lo, hi = ax.get_ylim()
        for (yv, xv), ty in zip(sorted(ends), spread([e[0] for e in sorted(ends)], 0.04 * (hi - lo))):
            ax.annotate(f"{yv:.3f}", (xv, yv), xytext=(xv + 0.02, ty), va="center", fontsize=8, color=INK2)
        style(ax, ylabel)
        ax.set_xlabel("relative depth (block l at (l + 1) / rows; final state at 1)", color=INK2)
    axes[0].legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.suptitle("M1: per-layer LDA, fit on the 20k train sample, scored on ProcessBench",
                 color=INK, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_m3(mech: Path, out: Path) -> Path:
    s = json.loads((mech / "m3_attention_summary.json").read_text())
    segs, tokens = s["segments"], list(s["tokens"])
    fig, axes = plt.subplots(2, (len(segs) + 1) // 2, figsize=(13, 6.5), sharex=True)
    for ax, seg in zip(axes.flat, segs):
        for name, color in zip(tokens, SERIES):
            y = s["tokens"][name]["mass"][seg]
            ax.plot(range(len(y)), y, color=color, lw=2, label=name)
        style(ax, "attention mass")
        ax.set_title(seg, color=INK, fontsize=10)
        ax.set_xlabel("layer", color=INK2)
    axes.flat[0].legend(frameon=False, labelcolor=INK)
    for ax in list(axes.flat)[len(segs):]:
        ax.axis("off")
    fig.suptitle(f"M3: attention from the read position, mean over heads and {s['n_items']} "
                 "PRM800K test items", color=INK, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figure", choices=["m1", "m3"])
    p.add_argument("--mech", type=Path, default=Path("results/read_token_v1/mech"))
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    fn = {"m1": plot_m1, "m3": plot_m3}[a.figure]
    print(fn(a.mech, a.out or a.mech / f"{a.figure}_layers.png"))


if __name__ == "__main__":
    main()
