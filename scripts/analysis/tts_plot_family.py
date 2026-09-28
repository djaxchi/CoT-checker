#!/usr/bin/env python3
"""Sprint 7's rules against the aggregators the test-time-scaling field uses.

Sprint 7 asked which of three rules to apply to N samples. The answer only
means something if the three were the right three, and they were not the
field's: the parallel-scaling literature aggregates with a *weighted* vote,
either by a verifier's reward (Lightman et al. 2023) or by a trace's own
confidence (DeepConf arXiv:2508.15260 Eq 8). This draws all of them on one
axis, from the same draws on the same problems, so the comparison is paired by
construction and the only thing that differs between curves is the rule.

Intervals are deliberately absent. Every curve here is a different rule applied
to the identical draws, so the per-curve interval is the wrong uncertainty to
look at; `tts_rule_contrast.py` bootstraps the paired difference, which is the
right one, and that is what the text should quote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROBE = "probe::step_tokens__transformer_d256_l2_f1024_h4__seed42::worst"
CONF = "conf::bottom10_group_w32"

# label, rule key, colour, linestyle, marker, z-order bump
SERIES = [
    ("best possible choice", "oracle", "#A8A8A8", "--", None, 0),
    ("self-consistency", "majority", "#8A8A8A", "-", "o", 1),
    ("verifier tie-break (ours)", f"tiebreak::{PROBE}", "#C0392B", "-", "o", 5),
    ("verifier weighted vote", f"wvote::{PROBE}", "#2A6FB5", "--", "s", 3),
    ("verifier best-of-N", f"rerank::{PROBE}", "#2A6FB5", ":", "^", 2),
    ("DeepConf weighted vote", f"wvote::{CONF}", "#E08214", "--", "D", 3),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frontier", type=Path,
                   default=Path("results/tts_roster_v1/frontier_v2.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/tts_roster_v1/figs/family_n_tokens.png"))
    p.add_argument("--datasets", nargs="+", default=["gsm8k", "math500"])
    a = p.parse_args()

    curves = json.loads(a.frontier.read_text())["curves"]
    fig, axes = plt.subplots(1, len(a.datasets), figsize=(12.2, 4.6))
    axes = [axes] if len(a.datasets) == 1 else list(axes)

    for ax, ds in zip(axes, a.datasets):
        rows = [r for r in curves if r["dataset"] == ds and r["split"] == "all"]
        ns = sorted({r["n"] for r in rows})
        for label, rule, colour, style, marker, z in SERIES:
            by_n = {r["n"]: r["accuracy"] for r in rows if r["rule"] == rule}
            if len(by_n) < len(ns):
                print(f"[skip] {ds}: {rule} missing at some budget")
                continue
            ax.plot(ns, [by_n[n] for n in ns], style, color=colour,
                    marker=marker, markersize=4, linewidth=2.0 if z == 5 else 1.4,
                    label=label, zorder=z)
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("samples drawn (N)")
        ax.set_xticks(ns)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        # Generated tokens beneath N, because the budget a practitioner spends
        # is tokens and the two axes are not proportional across datasets.
        tok = {r["n"]: r["tokens_mean"] for r in rows if r["rule"] == "majority"}
        sec = ax.secondary_xaxis(-0.18)
        sec.set_xticks(ns)
        sec.set_xticklabels([f"{tok[n] / 1000:.1f}k" for n in ns], fontsize=8)
        sec.set_xlabel("generated tokens per problem", fontsize=9)
        sec.spines["bottom"].set_visible(False)
        sec.tick_params(length=0)

    axes[0].set_ylabel("final-answer accuracy")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
