#!/usr/bin/env python3
"""Overlay per-token trajectories of matched fork steps (same problem + prefix).

The fork dataset pairs, for one shared reasoning prefix, a CORRECT next step (rating +1,
label 0) against an INCORRECT one (rating -1, label 1). This plot answers Djalil's
question directly: *given the same context, how does the model's per-token signal differ
between the correct and the incorrect continuation?*

Input is the ``{stem}_tokens.jsonl`` written by s3_token_incorrectness_extract.py run on
``forks_*_items.jsonl`` (so every row carries ``fork_id``/``role``/``label``). For each
selected fork we draw one figure with four stacked panels, x-axis = token position within
the step, one line per step, blue = correct / red = incorrect:

  1. P(step incorrect)  = sigmoid(probe score)      the deployed detector's readout
  2. confidence         = max softmax prob (p_top1)  how sure the model is of each token
  3. highest activation = max_i |h_i|                peak residual-stream dim (L{layer})
  4. # active dims      = #{i : |h_i| > tau}         count of strongly-active dims

A final aggregate figure averages the four signals over ALL two-sided forks on a common
normalized position axis (0 = first step token, 1 = last), correct vs incorrect, with a
+/- SEM band, so the systematic difference is visible beyond any single example.

Usage:
  python scripts/analysis/s3_fork_token_overlay_plot.py \
    --tokens runs/s3_fork_traj/qwen2_5_7b/forks_val_items_tokens.jsonl \
    --out runs/s3_fork_traj/qwen2_5_7b/plots --layer 28 --n_forks 12
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CORRECT_C = "#2166ac"   # blue  (label 0, rating +1)
INCORRECT_C = "#d6301d"  # red   (label 1, rating -1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def load_forks(tokens_path: Path, layer: int) -> dict:
    """Group token rows into {fork_id: {uid: step}}.  Each step keeps its per-token
    metric arrays ordered by tok_pos, plus label and a prefix/problem snippet."""
    need_probe = f"probe_score_l{layer}"
    need_absmax = f"hidden_absmax_l{layer}"
    need_nact = f"hidden_nact_l{layer}"
    forks: dict = defaultdict(lambda: defaultdict(lambda: {
        "tok_pos": [], "pinc": [], "conf": [], "absmax": [], "nact": [],
        "label": None, "role": None, "problem_id": None,
    }))
    with tokens_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            fid = r.get("fork_id")
            if fid is None:
                continue  # heldout row, not a fork
            st = forks[fid][r["uid"]]
            st["tok_pos"].append(r["tok_pos"])
            st["pinc"].append(sigmoid(r[need_probe]))
            st["conf"].append(r["p_top1"])
            st["absmax"].append(r[need_absmax])
            st["nact"].append(r[need_nact])
            st["label"] = int(r["label"])
            st["role"] = r.get("role")
            st["problem_id"] = r.get("problem_id")

    # sort each step's arrays by token position and convert to numpy
    out: dict = {}
    for fid, steps in forks.items():
        clean = {}
        for uid, st in steps.items():
            order = np.argsort(st["tok_pos"])
            clean[uid] = {
                "label": st["label"], "role": st["role"],
                "problem_id": st["problem_id"],
                "pinc": np.asarray(st["pinc"])[order],
                "conf": np.asarray(st["conf"])[order],
                "absmax": np.asarray(st["absmax"])[order],
                "nact": np.asarray(st["nact"])[order],
            }
        out[fid] = clean
    return out


def two_sided(forks: dict) -> list:
    """Fork ids that have at least one correct (label 0) and one incorrect (label 1) step."""
    keep = []
    for fid, steps in forks.items():
        labels = {s["label"] for s in steps.values()}
        if 0 in labels and 1 in labels:
            keep.append(fid)
    return keep


def fork_separation(steps: dict) -> float:
    """mean P(incorrect) over incorrect steps minus over correct steps (readout gap)."""
    inc = [s["pinc"].mean() for s in steps.values() if s["label"] == 1]
    cor = [s["pinc"].mean() for s in steps.values() if s["label"] == 0]
    if not inc or not cor:
        return float("-inf")
    return float(np.mean(inc) - np.mean(cor))


PANELS = [
    ("pinc", "P(step incorrect)\n= sigmoid(probe score)", (0.0, 1.0), 0.5),
    ("conf", "model confidence\n= max softmax prob", (0.0, 1.0), None),
    ("absmax", "highest activation\n= max_i |h_i|", None, None),
    ("nact", "# strongly-active dims\n= #{|h_i| > tau}", None, None),
]


def plot_fork(fid: str, steps: dict, layer: int, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for ax, (key, ylabel, ylim, hline) in zip(axes, PANELS):
        for uid, s in steps.items():
            c = INCORRECT_C if s["label"] == 1 else CORRECT_C
            x = np.arange(s[key].shape[0])
            ax.plot(x, s[key], color=c, alpha=0.8, lw=1.4,
                    marker="o", ms=2.5)
        if hline is not None:
            ax.axhline(hline, color="0.4", ls="--", lw=0.8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("token position within step (0 = first generated token)")
    n_cor = sum(s["label"] == 0 for s in steps.values())
    n_inc = sum(s["label"] == 1 for s in steps.values())
    handles = [plt.Line2D([], [], color=CORRECT_C, lw=2,
                          label=f"correct step (+1)  n={n_cor}"),
               plt.Line2D([], [], color=INCORRECT_C, lw=2,
                          label=f"incorrect step (-1)  n={n_inc}")]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)
    sep = fork_separation(steps)
    fig.suptitle(f"fork {fid}\nsame problem+prefix, correct vs incorrect next step  "
                 f"(L{layer}; readout gap dP(inc)={sep:+.2f})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def resample(y: np.ndarray, n: int) -> np.ndarray:
    """Linear-resample a per-token trajectory onto n points over normalized position."""
    y = np.asarray(y, dtype=np.float64)
    if y.shape[0] == 1:
        return np.full(n, y[0])
    xp = np.linspace(0.0, 1.0, y.shape[0])
    return np.interp(np.linspace(0.0, 1.0, n), xp, y)


def plot_aggregate(forks: dict, fids: list, layer: int, out_path: Path,
                   n_grid: int = 20) -> dict:
    grid = np.linspace(0.0, 1.0, n_grid)
    stacks = {key: {0: [], 1: []} for key, _, _, _ in PANELS}
    for fid in fids:
        for s in forks[fid].values():
            for key, _, _, _ in PANELS:
                stacks[key][s["label"]].append(resample(s[key], n_grid))

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    summary: dict = {}
    for ax, (key, ylabel, ylim, hline) in zip(axes, PANELS):
        for lab, c, name in ((0, CORRECT_C, "correct"), (1, INCORRECT_C, "incorrect")):
            arr = np.asarray(stacks[key][lab])           # (n_steps, n_grid)
            m = arr.mean(axis=0)
            sem = arr.std(axis=0) / np.sqrt(max(arr.shape[0], 1))
            ax.plot(grid, m, color=c, lw=2.2, label=f"{name}  n={arr.shape[0]}")
            ax.fill_between(grid, m - sem, m + sem, color=c, alpha=0.2)
            summary[f"{key}_{name}_first"] = float(m[0])
            summary[f"{key}_{name}_last"] = float(m[-1])
            summary[f"{key}_{name}_mean"] = float(m.mean())
        if hline is not None:
            ax.axhline(hline, color="0.4", ls="--", lw=0.8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("normalized position within step (0 = first token, 1 = last)")
    fig.suptitle(f"mean per-token trajectory over {len(fids)} matched forks (L{layer})\n"
                 "correct vs incorrect next step, +/- SEM band", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=28, help="probe/hidden layer to plot")
    ap.add_argument("--n_forks", type=int, default=12, help="how many forks to draw")
    ap.add_argument("--order", choices=["separation", "first", "random"],
                    default="separation",
                    help="which forks to draw: clearest readout gap / file order / random")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    forks = load_forks(args.tokens, args.layer)
    fids = two_sided(forks)
    if not fids:
        raise SystemExit(f"no two-sided forks in {args.tokens}; is this a fork extract?")
    print(f"[fork-overlay] {len(forks)} forks, {len(fids)} two-sided (>=1 correct & "
          f">=1 incorrect step)")

    if args.order == "separation":
        fids.sort(key=lambda f: fork_separation(forks[f]), reverse=True)
    elif args.order == "random":
        np.random.default_rng(args.seed).shuffle(fids)

    written = []
    for fid in fids[:args.n_forks]:
        safe = fid.replace("::", "_").replace("/", "_")
        p = args.out / f"fork_{safe}_L{args.layer}.png"
        plot_fork(fid, forks[fid], args.layer, p)
        written.append(p)

    agg_path = args.out / f"aggregate_forks_L{args.layer}.png"
    summary = plot_aggregate(forks, fids, args.layer, agg_path)
    written.append(agg_path)
    (args.out / f"aggregate_forks_L{args.layer}_summary.json").write_text(
        json.dumps({"n_two_sided_forks": len(fids), **summary}, indent=2))

    print(f"[fork-overlay] wrote {len(written)} figures to {args.out}")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
