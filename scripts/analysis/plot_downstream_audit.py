#!/usr/bin/env python3
"""Figures for the fixed-grader, exact-tie downstream audit and held-out repairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.audit_instruct_downstream import BOUNDARY, D512, PRM, STEPSTATS, interval

COLORS = {"10": "#147D92", "07": "#CB6232"}
LABELS = {"math500": "MATH-500", "gsm8k": "GSM8K"}


def error(ax: plt.Axes, metric: dict, y: float, color: str, marker: str = "o") -> None:
    m, (lo, hi) = metric["mean"], metric["ci95"]
    ax.errorbar(m, y, xerr=[[m-lo], [hi-m]], fmt=marker, color=color, capsize=3, ms=5)


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT/"results/instruct_downstream_v1/audit_v3_spacing")
    a = p.parse_args()
    data = json.loads((a.root/"audit.json").read_text())["results"]
    out = a.root/"figs"
    out.mkdir(exist_ok=True)
    paths = []

    def save(fig: plt.Figure, name: str) -> None:
        path = out/(name+".png")
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig); paths.append(str(path.resolve()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    cells = [(PRM, "PRM 7B"), (BOUNDARY, "Boundary MLP"), (STEPSTATS, "Step-stats MLP"), (D512, "Transformer d512")]
    for ax, (ds, title) in zip(axes, LABELS.items()):
        for t, offset in [("10", -.10), ("07", .10)]:
            for i, (cell, name) in enumerate(cells):
                metric = data[ds][t]["scorer_diagnostics"][cell]["strict_minority_correct"]["auc"]
                error(ax, metric, i+offset, COLORS[t])
        n10 = data[ds]["10"]["cases"]["strict_minority_correct"]
        n07 = data[ds]["07"]["cases"]["strict_minority_correct"]
        ax.set_title(f"{title} | {n10} / {n07} strict-minority problems", loc="left")
        ax.axvline(.5, color="#555555", lw=1)
        ax.set_yticks(range(4), [n for _, n in cells]); ax.invert_yaxis()
        ax.set_xlim(.25, .9); ax.set_xlabel("Within-problem AUROC"); clean(ax)
    axes[0].invert_yaxis()
    fig.suptitle("Minority-answer ranking: uncertainty matters", x=.16, ha="left", fontsize=17)
    fig.text(.16, .09, "Blue: temperature 1.0. Orange: temperature 0.7. Bars: 95% bootstrap over problems.", fontsize=10)
    fig.text(.16, .035, "Correct answers must lose the plurality outright; vote ties are separate. Spacing-corrected grader.", fontsize=10)
    fig.subplots_adjust(left=.16, right=.98, top=.82, bottom=.23, wspace=.12)
    save(fig, "strict_minority_auc")

    specs = [("fixed", f"{BOUNDARY}|worst|rerank", "Boundary MLP, original"),
             ("fixed", f"{PRM}|worst|rerank", "PRM, original"),
             ("crossfit", "probe_aggregation_rerank", "Select probe + aggregation"),
             ("crossfit", "probe_conservative", "Select conservative probe rule"),
             ("pairwise_repair", "length_only", "Train ranker: length only"),
             ("pairwise_repair", "scores_only", "Train ranker: probe scores"),
             ("pairwise_repair", "scores_length_conf", "+ length + confidence"),
             ("pairwise_repair", "scores_length_conf_votes", "+ vote share")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), sharey=True)
    for ax, (ds, title) in zip(axes, LABELS.items()):
        for t, offset in [("10", -.12), ("07", .12)]:
            for i, (family, name, _) in enumerate(specs):
                metric = data[ds][t]["policies"][name]["gain_pp"] if family == "fixed" else data[ds][family][name]["pools"][t]["gain_pp"]
                error(ax, metric, i+offset, COLORS[t])
        ax.set_title(title, loc="left"); ax.axvline(0, color="#555555", lw=1)
        ax.axhline(1.5, color="#999999", lw=.8, ls=":")
        ax.set_yticks(range(len(specs)), [x[2] for x in specs])
        ax.set_xlabel("N=10 gain over majority vote (pp)"); clean(ax)
    axes[0].invert_yaxis()
    fig.suptitle("Held-out repairs do not establish a main-pool probe gain", x=.26, ha="left", fontsize=17)
    fig.text(.26, .075, "Blue: temperature 1.0. Orange: temperature 0.7. First two rows are fixed reference policies.", fontsize=10)
    fig.text(.26, .045, "Other rows hold out whole questions across both pools. Exploratory; no independent new test set.", fontsize=10)
    fig.text(.26, .015, "95% problem-bootstrap bars condition on selected/fitted fold models; they omit training/selection uncertainty.", fontsize=9)
    fig.subplots_adjust(left=.26, right=.98, top=.87, bottom=.18, wspace=.12)
    save(fig, "held_out_repairs")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, t in zip(axes, ["10", "07"]):
        r = data["gsm8k"][t]
        reference = r["prm_traces_per_problem"][f"{PRM}|worst|rerank"]
        for gate, label, color, marker in [("vote_only", "Vote margin gate", "#147D92", "o"),
                                          ("probe_disagrees", "Vote margin + probe disagreement", "#CB6232", "^")]:
            keys = [f"cascade|{gate}|margin{m}|rerank" for m in ["0", "0.2", "0.4", "1"]]
            x = [100*r["prm_traces_per_problem"][k]/reference for k in keys]
            y = [r["policies"][k]["gain_pp"]["mean"] for k in keys]
            ax.plot(x, y, marker+"-", color=color, label=label)
        full = r["policies"][f"{PRM}|worst|rerank"]["gain_pp"]["mean"]
        ax.axhline(full, color="#777777", ls="--", lw=1)
        ax.plot(0, 0, "s", color="#222222")
        ax.set_title(f"GSM8K | temperature {int(t)/10:.1f}", loc="left")
        ax.set_xlabel("PRM trace reads (% of ambiguity-only PRM)"); clean(ax)
        ax.set_xlim(-3, 104)
    axes[0].set_ylabel("N=10 gain over majority vote (pp)")
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle("Probe triage trades some PRM accuracy for fewer reads", x=.08, ha="left", fontsize=17)
    fig.text(.08, .08, "Fair reference already skips unanimous answer pools. Fixed gates shown, not selected on held-out performance.", fontsize=10)
    fig.text(.08, .035, "Costs count PRM trace reads, excluding probe overhead and token-length differences. These are not measured speedups.", fontsize=10)
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.22, wspace=.12)
    save(fig, "prm_triage_tradeoff")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    cells2 = cells + [("confidence", "Token confidence")]
    for ax, (ds, title) in zip(axes, LABELS.items()):
        for t, offset in [("10", -.1), ("07", .1)]:
            for i, (cell, _) in enumerate(cells2):
                error(ax, data[ds][t]["exact_N2"][cell]["gain_pp"], i+offset, COLORS[t])
        ax.set_title(title, loc="left"); ax.axvline(0, color="#555555", lw=1)
        ax.set_yticks(range(5), [n for _, n in cells2]); ax.set_xlabel("N=2 gain over majority vote (pp)"); clean(ax)
    axes[0].invert_yaxis()
    fig.suptitle("Small-pool tie-breaking remains useful", x=.16, ha="left", fontsize=17)
    fig.text(.16, .08, "All 45 pairs per problem, uniform score ties. Blue: temperature 1.0. Orange: temperature 0.7.", fontsize=10)
    fig.text(.16, .035, "95% paired problem-bootstrap intervals versus majority. Overlapping bars do not establish probe-PRM equivalence.", fontsize=10)
    fig.subplots_adjust(left=.16, right=.98, top=.84, bottom=.23, wspace=.12)
    save(fig, "exact_pair_gains")

    rows = [json.loads(line) for line in (a.root/"per_problem.jsonl").read_text().splitlines()]
    comparisons = {}
    for t in ["10", "07"]:
        subset = [r for r in rows if r["dataset"] == "gsm8k" and r["temperature"] == t]
        key = "cascade|probe_disagrees|margin1|rerank"
        baseline = f"{PRM}|worst|rerank"
        d = np.array([100*(r["policies"][key]-r["policies"][baseline]) for r in subset])
        r = data["gsm8k"][t]
        comparisons[t] = {"vs_prm_pp": interval(d),
                          "prm_trace_reads_per_problem": r["prm_traces_per_problem"][key],
                          "fraction_prm_reads_saved": 1-r["prm_traces_per_problem"][key]/r["prm_traces_per_problem"][baseline]}
    (a.root/"deployment_tradeoffs.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    (out/"manifest.json").write_text(json.dumps({"figures": paths}, indent=2)+"\n")
    print(json.dumps(comparisons, indent=2)); print("\n".join(paths))


if __name__ == "__main__":
    main()
