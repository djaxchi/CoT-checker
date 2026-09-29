#!/usr/bin/env python3
"""Plot saved downstream contrasts without rerunning inference or bootstraps.

Usage: python scripts/analysis/plot_instruct_downstream.py
Intervals are the existing pointwise paired question-cluster bootstrap CIs.
The 16-cell analysis is a post-hoc sensitivity, not the registered main test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.instruct_downstream_summary import spearman  # noqa: E402

PRM = "probe::prm_qwen25_math_7b::worst"
CONF = "conf::mean_token_conf"
NS = [2, 3, 4, 6, 8, 10]
DATASETS = {"math500": "MATH-500", "gsm8k": "GSM8K"}
RULES = {"tiebreak": "Break vote ties", "rerank": "Best-of-N", "wvote": "Weighted vote"}
COLORS = ["#147D92", "#CB6232", "#7762A8", "#6F7782"]


def contrast(row: dict) -> tuple[float, list[float]]:
    """Convert baseline-minus-rule to percentage-point rule-minus-baseline."""
    return -100 * row["delta"], [-100 * row["ci95"][1], -100 * row["ci95"][0]]


def load_checked(root: Path, temperature: str) -> dict:
    """Check summary signs, intervals and repeated rows against raw contrasts."""
    data = json.loads((root / f"summary_t{temperature}.json").read_text())
    seen = {}
    for path in sorted((root / "parts").glob(f"contrast_t{temperature}_*.json")):
        for row in json.loads(path.read_text()):
            key = f"{row['dataset']}|{row['n']}|{row['rule']}"
            lift, ci = contrast(row)
            values = [lift, *ci, row["rule_acc"], row["baseline_acc"], row["n_problems"]]
            if key in seen and not np.allclose(values, seen[key], atol=1e-10, rtol=0):
                raise ValueError(f"Conflicting repeated contrast: {key}")
            seen[key] = values
            expected = [100 * data["lift"][key], *np.multiply(data["ci95"][key], 100),
                        data["accuracy"][key], data["majority"][f"{row['dataset']}|{row['n']}"]]
            if not np.allclose(values[:5], expected, atol=1e-10, rtol=0):
                raise ValueError(f"Summary differs from raw contrast: {key}")
            if not np.isclose(lift, 100 * (row["rule_acc"] - row["baseline_acc"])):
                raise ValueError(f"Incorrect lift sign: {key}")
    if set(seen) != set(data["lift"]):
        raise ValueError("Raw contrasts and summary have different endpoint coverage")
    return data


def selected(data: dict) -> list[tuple[str, str]]:
    def scorer(rep: str, learner: str) -> str:
        return next(c["scorer"] for c in data["cells"] if c["rep"] == rep and c["learner"] == learner)
    return [(PRM, "PRM 7B"), (scorer("boundary_stats", "mlp:h1024x2"), "Boundary MLP (1024 x 2)"),
            (scorer("step_tokens", "transformer:d512,l2,f2048,h8"), "Transformer d512"),
            (CONF, "Mean token confidence")]


def endpoint(data: dict, ds: str, n: int, rule: str, scorer: str) -> tuple[float, np.ndarray]:
    key = f"{ds}|{n}|{rule}::{scorer}"
    return 100 * data["lift"][key], 100 * np.asarray(data["ci95"][key])


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.17)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, out: Path, name: str, paths: list[str]) -> None:
    path = out / f"{name}.png"
    fig.savefig(path, dpi=180, facecolor="white")
    paths.append(str(path.resolve()))
    plt.close(fig)


def scaling(data: dict, temp: str, out: Path, paths: list[str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for row, (ds, title) in enumerate(DATASETS.items()):
        for col, (rule, label) in enumerate(RULES.items()):
            ax = axes[row, col]
            ax.axhline(0, color="#333333", lw=1)
            for (sc, name), color in zip(selected(data), COLORS):
                vals = [endpoint(data, ds, n, rule, sc) for n in NS]
                y = np.array([v[0] for v in vals]); bounds = np.array([v[1] for v in vals])
                ax.plot(NS, y, "o-", color=color, ms=4, label=name, lw=1.8)
                ax.fill_between(NS, bounds[:, 0], bounds[:, 1], color=color, alpha=.09)
            ax.set_title(f"{title} | {label}", loc="left", fontsize=12)
            ax.set_xticks(NS); clean(ax)
            ax.set_ylim(-7, 3.3)
            if col == 0:
                ax.set_ylabel("Gain over majority vote (pp)")
            if row == 1:
                ax.set_xlabel("Number of sampled solutions, N")
    fig.suptitle(f"Instruct selection gains | temperature {temp}", x=.06, ha="left", fontsize=19)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.51, .94), ncol=4, frameon=False)
    fig.text(.06, .035, "Bands: pointwise 95% paired question-cluster bootstrap intervals. Zero: majority vote on the same draws.", fontsize=10)
    fig.text(.06, .012, "Boundary MLP highlighted after inspecting downstream results. Equal N does not imply equal compute. Lines connect evaluated budgets.", fontsize=9, color="#555555")
    fig.subplots_adjust(top=.84, bottom=.12, left=.07, right=.98, hspace=.30, wspace=.14)
    save(fig, out, "selection_gains_t" + temp.replace(".", ""), paths)


def transfer_metrics(data: dict) -> dict:
    result = {}
    for subset_name, cells in [("all_19", data["cells"]),
                               ("excluding_step_delta_16", [c for c in data["cells"] if c["rep"] != "step_delta"])]:
        result[subset_name] = {
            field: spearman([c[field] for c in cells],
                            [endpoint(data, "math500", 4, "rerank", c["scorer"])[0] for c in cells])
            for field in ["calib20_seed", "calib20_mean"]}
    return result


def rank_transfer(pools: dict, out: Path, paths: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col", sharey=True)
    for row, (tag, data) in enumerate(pools.items()):
        metrics = transfer_metrics(data)
        for col, (field, label) in enumerate([("calib20_seed", "Chosen seed F1"), ("calib20_mean", "Seed-averaged F1")]):
            ax = axes[row, col]
            for c in data["cells"]:
                broken = c["rep"] == "step_delta"
                y, _ = endpoint(data, "math500", 4, "rerank", c["scorer"])
                ax.scatter(c[field], y, marker="x" if broken else "o", s=50,
                           color="#B74B4B" if broken else "#9BA7AF", alpha=.85)
            for sc, name, color in [(selected(data)[1][0], "Boundary MLP", COLORS[1]),
                                     (selected(data)[2][0], "d512", COLORS[2])]:
                cell = next(c for c in data["cells"] if c["scorer"] == sc)
                y, _ = endpoint(data, "math500", 4, "rerank", sc)
                ax.scatter(cell[field], y, s=65, color=color, zorder=4)
                ax.annotate(name, (cell[field], y), xytext=(-8, 9), textcoords="offset points", ha="right", fontsize=9, color=color)
            ax.text(.03, .96, f"Spearman: all 19 = {metrics['all_19'][field]:+.2f}\nWithout step_delta = {metrics['excluding_step_delta_16'][field]:+.2f}",
                    transform=ax.transAxes, va="top", fontsize=10)
            ax.set_ylim(-4.85, .95)
            ax.set_title(f"Temperature {int(tag)/10:.1f} | {label}", loc="left")
            if col == 0:
                ax.set_ylabel("MATH-500 best-of-4 gain (pp)")
            if row == 1:
                ax.set_xlabel("ProcessBench calib-20 F1")
            ax.axhline(0, color="#333333", lw=1); clean(ax)
    fig.suptitle("ProcessBench rank transfer depends on the pool", x=.08, ha="left", fontsize=18)
    fig.text(.08, .045, "Crosses: three step_delta cells with reported generation-state failures. Excluding them is a post-hoc sensitivity.", fontsize=10)
    fig.text(.08, .018, "Both columns use the same chosen checkpoint downstream. Seed-averaged F1 is not seed-averaged downstream performance.", fontsize=10)
    fig.subplots_adjust(top=.89, bottom=.14, left=.08, right=.98, hspace=.28, wspace=.13)
    save(fig, out, "processbench_rank_transfer", paths)


def cell_label(c: dict) -> str:
    learner = c["learner"].replace("mlp:h", "MLP ").replace("x2", " x 2")
    if learner.startswith("transformer"):
        learner = "Transformer " + learner.split(":")[1].split(",")[0]
    return f"{c['rep']} / {learner}"


def all_cells(data: dict, out: Path, paths: list[str]) -> None:
    cells = sorted([c for c in data["cells"] if c["rep"] != "step_delta"],
                   key=lambda c: -endpoint(data, "math500", 4, "rerank", c["scorer"])[0])
    failed = [c for c in data["cells"] if c["rep"] == "step_delta"]
    rows = [(PRM, "PRM 7B"), (CONF, "Mean token confidence"),
            ("conf::bottom10_group_w32", "Bottom-10 token confidence")]
    rows += [(c["scorer"], cell_label(c)) for c in cells + failed]
    endpoints = [(2, "tiebreak"), (4, "tiebreak"), (10, "tiebreak"),
                 (4, "rerank"), (10, "rerank"), (10, "wvote")]
    labels = ["Tie\nN=2", "Tie\nN=4", "Tie\nN=10", "Best\nN=4", "Best\nN=10", "Weight\nN=10"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        matrix = np.array([[endpoint(data, ds, n, rule, sc)[0] for n, rule in endpoints] for sc, _ in rows])
        im = ax.imshow(matrix, cmap="RdBu", vmin=-4, vmax=4, aspect="auto")
        for i, (sc, _) in enumerate(rows):
            for j, (n, rule) in enumerate(endpoints):
                y, ci = endpoint(data, ds, n, rule, sc)
                star = "*" if ci[0] > 0 or ci[1] < 0 else ""
                ax.text(j, i, f"{y:+.2f}{star}", ha="center", va="center", fontsize=9,
                        color="white" if abs(y) > 2.6 else "#222222")
        ax.set_xticks(range(6), labels); ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)), [label for _, label in rows] if ds == "math500" else [])
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=14, pad=38, loc="left")
        ax.axhline(2.5, color="#444444", lw=1.3)
        ax.axhline(len(rows)-3.5, color="#B74B4B", lw=2)
    fig.suptitle("All verifiers | temperature 1.0 | gain over majority vote (pp)", x=.03, ha="left", fontsize=18)
    cax = fig.add_axes([.90, .18, .015, .60])
    fig.colorbar(im, cax=cax, extend="both", label="Gain (pp), colors saturate at +/-4")
    fig.text(.03, .045, "* Pointwise 95% paired CI excludes zero; no multiple-comparison correction. Last three rows: reported step_delta failures.", fontsize=10)
    fig.text(.03, .02, "Probe rows sorted by MATH-500 best-of-4 gain, with failures separated. Numbers show unclipped effects, including the -63.97 pp collapse.", fontsize=10)
    fig.subplots_adjust(left=.29, right=.88, top=.87, bottom=.09, wspace=.08)
    save(fig, out, "all_verifier_gains_t10", paths)


def headroom(pools: dict, out: Path, paths: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="col")
    for row, (tag, data) in enumerate(pools.items()):
        for col, (ds, title) in enumerate(DATASETS.items()):
            ax = axes[row, col]
            majority = [100 * data["majority"][f"{ds}|{n}"] for n in NS]
            oracle = [100 * data["accuracy"][f"{ds}|{n}|oracle"] for n in NS]
            ax.fill_between(NS, majority, oracle, color="#DAE6E8", alpha=.55)
            ax.plot(NS, oracle, "--", color="#536975", label="Oracle: any correct sample")
            ax.plot(NS, majority, "o-", color="#202C36", label="Majority vote")
            for sc, rule, label, color in [(PRM, "rerank", "PRM best-of-N", COLORS[0]),
                                           (selected(data)[1][0], "tiebreak", "Boundary MLP tie-break", COLORS[1])]:
                y = [100 * data["accuracy"][f"{ds}|{n}|{rule}::{sc}"] for n in NS]
                ax.plot(NS, y, "o-", ms=4, color=color, label=label)
            ax.set_title(f"{title} | temperature {int(tag)/10:.1f}", loc="left")
            ax.text(.97, .04, f"N=10 oracle gap: {oracle[-1]-majority[-1]:.2f} pp", transform=ax.transAxes, ha="right", fontsize=10)
            ax.set_xticks(NS); clean(ax)
            if col == 0:
                ax.set_ylabel("Answer accuracy (%)")
            if row == 1:
                ax.set_xlabel("Number of sampled solutions, N")
    fig.suptitle("The pools still contain recoverable answers", x=.08, ha="left", fontsize=18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .94), ncol=2, frameon=False)
    fig.text(.08, .02, "Shading: oracle minus majority, an upper bound on selection gains within the sampled pool. Point estimates; not a compute-matched comparison.", fontsize=9)
    fig.subplots_adjust(top=.81, bottom=.11, left=.08, right=.98, hspace=.28, wspace=.15)
    save(fig, out, "accuracy_and_oracle_headroom", paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/instruct_downstream_v1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.root / "figs/analysis_v2"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "medium"})
    pools = {tag: load_checked(args.root, tag) for tag in ["10", "07"]}
    paths = []
    for tag, data in pools.items():
        scaling(data, f"{int(tag)/10:.1f}", out, paths)
    rank_transfer(pools, out, paths)
    all_cells(pools["10"], out, paths)
    headroom(pools, out, paths)
    sources = sorted(args.root.glob("summary_t*.json")) + sorted((args.root / "parts").glob("*.json"))
    metrics = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "rank_transfer": {tag: transfer_metrics(d) for tag, d in pools.items()},
               "figures": paths,
               "notes": ["No new bootstrap intervals computed; plotted existing paired pointwise CIs.",
                         "No direct paired PRM-versus-probe inference from aggregate contrasts.",
                         "16-cell exclusion is post-hoc; all 19 cells remain in the main analysis.",
                         "Boundary MLP is a post-hoc downstream highlight, not a prespecified winner."]}
    (out / "analysis_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("\n".join(paths))
    print(json.dumps(metrics["rank_transfer"], indent=2))


if __name__ == "__main__":
    main()
