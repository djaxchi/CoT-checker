#!/usr/bin/env python3
"""Visualise PTB experiment results.

Produces five figures:
  fig1_ms_performance.png     -- Math-Shepherd metrics (in-distribution)
  fig2_pb_transfer.png        -- ProcessBench transfer metrics (out-of-distribution)
  fig3_step_lda.png           -- Per-step LDA score: mean ± std for correct vs incorrect
  fig4_lda_geometry.png       -- KDE of LDA scores per class (overall class separability)
  fig5_lda_trajectories.png   -- Individual solution LDA scores across steps

Four key representations (figs 3-5):
  dense   = raw backbone h_k             (--dense-eval)
  ssae_z  = SSAE sparse latents          (--ssae-eval, optional)
  ptb     = PTB bottleneck (no_l1)       (--rep-dir / ptb_no_l1_ms_eval.npz)
  delta   = dense transition delta h_k   (--rep-dir / dense_delta_ms_eval.npz)

SSAE note: if eval_held_out.npz lacks step_positions, borrow them from
--dense-eval (valid when both files cover the same ordered evaluation split).
Pass --no-ssae-meta to disable this.

Usage:
    python scripts/plot_ptb_results.py \\
        --results      results/ptb_robust_probes/summary_results.json \\
        --rep-dir      /scratch/cot-checker/probe_data/ptb_representations_robust \\
        --dense-eval   /store/probe_data/dense_eval_held_out.npz \\
        --ssae-eval    /store/probe_data/eval_held_out.npz \\
        --output-dir   results/ptb_figures
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------
CORRECT_COL   = "#1565C0"   # dark blue
INCORRECT_COL = "#C62828"   # dark red
FILL_ALPHA    = 0.18

# All-representations order and styling (figs 1-2)
ALL_REP_ORDER = [
    "dense_h", "dense_delta", "dense_concat",
    "random_bln",
    "ptb_no_l1", "ptb_dwa_calibrated", "ptb_active_fraction", "ptb_topk",
]
ALL_REP_LABELS = {
    "dense_h":             "Dense h_k",
    "dense_delta":         "Dense Δh",
    "dense_concat":        "Dense [h;Δh]",
    "random_bln":          "Random proj\n(on SSAE z)",
    "ptb_no_l1":           "PTB (no L1)",
    "ptb_dwa_calibrated":  "PTB (DWA cal.)",
    "ptb_active_fraction": "PTB (act. frac.)",
    "ptb_topk":            "PTB (TopK)",
}
ALL_REP_COLORS = {
    "dense_h":             "#546E7A",
    "dense_delta":         "#78909C",
    "dense_concat":        "#90A4AE",
    "random_bln":          "#EF6C00",
    "ptb_no_l1":           "#2E7D32",
    "ptb_dwa_calibrated":  "#43A047",
    "ptb_active_fraction": "#66BB6A",
    "ptb_topk":            "#A5D6A7",
}

# The four representations for detailed figures (3-5)
KEY_REPS   = ["dense", "ssae_z", "ptb", "delta"]
KEY_LABELS = {
    "dense":  "Dense h_k\n(backbone hidden state)",
    "ssae_z": "SSAE z_k\n(sparse autoencoder features)",
    "ptb":    "PTB (no L1)\n(transition bottleneck)",
    "delta":  "Dense Δh\n(transition delta)",
}
KEY_COLORS = {
    "dense":  "#546E7A",
    "ssae_z": "#6A1B9A",
    "ptb":    "#2E7D32",
    "delta":  "#0D47A1",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(
    path: str | Path,
    n_max: int | None = None,
    seed: int = 0,
    meta_src: str | Path | None = None,
) -> dict | None:
    """Load latents + metadata from an npz file.

    If step_positions is missing and meta_src is supplied, borrows it from
    meta_src (valid when both files are aligned over the same eval split).
    """
    p = Path(path)
    if not p.exists():
        print(f"  MISSING: {p}")
        return None
    d = np.load(p)
    h = d["latents"].astype(np.float32)
    y_key = next((k for k in ("correctness", "step_labels") if k in d.files), None)
    if y_key is None:
        print(f"  WARNING: no correctness key in {p.name}")
        return None
    y = d[y_key].astype(int)

    sol_ids  = d["solution_ids"].astype(int)  if "solution_ids"  in d.files else None
    step_pos = d["step_positions"].astype(int) if "step_positions" in d.files else None

    # Try to borrow step metadata from a reference file
    if (sol_ids is None or step_pos is None) and meta_src is not None:
        mp = Path(meta_src)
        if mp.exists():
            m = np.load(mp)
            if len(m["latents"]) == len(h):
                if sol_ids  is None and "solution_ids"  in m.files:
                    sol_ids  = m["solution_ids"].astype(int)
                if step_pos is None and "step_positions" in m.files:
                    step_pos = m["step_positions"].astype(int)
                print(f"  Borrowed step metadata for {p.name} from {mp.name}")
            else:
                print(f"  WARNING: {p.name} ({len(h)}) vs {mp.name} ({len(m['latents'])}) "
                      "length mismatch -- cannot borrow step metadata")

    if n_max is not None and len(h) > n_max:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(h), n_max, replace=False)
        idx.sort()
        h = h[idx]; y = y[idx]
        if sol_ids  is not None: sol_ids  = sol_ids[idx]
        if step_pos is not None: step_pos = step_pos[idx]

    return {"h": h, "y": y, "sol_ids": sol_ids, "step_pos": step_pos}


# ---------------------------------------------------------------------------
# LDA projection helper
# ---------------------------------------------------------------------------

def _lda_scores(h: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Project h onto the Fisher LDA axis. Correct class (y=1) gets higher score.

    Falls back to difference-of-means direction if LDA fails (e.g. rank-deficient).
    """
    try:
        lda = LinearDiscriminantAnalysis(n_components=1, solver="svd")
        scores = lda.fit_transform(h, y).ravel()
    except Exception as exc:
        print(f"    LDA fallback (diff-of-means): {exc}")
        d = h[y == 1].mean(0) - h[y == 0].mean(0)
        norm = np.linalg.norm(d)
        if norm < 1e-10:
            return np.zeros(len(h))
        scores = h @ (d / norm)

    if scores[y == 1].mean() < scores[y == 0].mean():
        scores = -scores
    return scores


# ---------------------------------------------------------------------------
# Figure 1: MS Performance
# ---------------------------------------------------------------------------

def plot_ms_performance(rows: list[dict], out_path: Path) -> None:
    rows = sorted(
        [r for r in rows if r["label"] in ALL_REP_ORDER],
        key=lambda r: ALL_REP_ORDER.index(r["label"]),
    )
    labels   = [ALL_REP_LABELS.get(r["label"], r["label"]) for r in rows]
    ms_mf1   = [r["ms_macro_f1"] for r in rows]
    ms_std   = [r.get("ms_mf1_std", 0.0) for r in rows]
    ms_auroc = [r["ms_auroc"] if r["ms_auroc"] == r["ms_auroc"] else 0.0 for r in rows]
    colors   = [ALL_REP_COLORS.get(r["label"], "#999") for r in rows]
    n = len(rows)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Math-Shepherd (in-distribution) probe performance", fontsize=13, fontweight="bold")

    for ax, vals, stds, title, ylim in [
        (axes[0], ms_mf1,   ms_std, "Macro-F1 (balanced threshold)", (0.60, 0.80)),
        (axes[1], ms_auroc, [0]*n,  "AUROC",                         (0.70, 0.90)),
    ]:
        bars = ax.bar(x, vals, width=0.6, color=colors, edgecolor="white", linewidth=0.8,
                      yerr=stds, capsize=3, error_kw={"ecolor": "#444", "lw": 1.2})
        ax.axhline(0.5, color="#aaa", lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        for bar, val, std in zip(bars, vals, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: PB Transfer
# ---------------------------------------------------------------------------

def plot_pb_transfer(rows: list[dict], out_path: Path) -> None:
    rows = sorted(
        [r for r in rows if r["label"] in ALL_REP_ORDER],
        key=lambda r: ALL_REP_ORDER.index(r["label"]),
    )
    labels = [ALL_REP_LABELS.get(r["label"], r["label"]) for r in rows]
    pb_f1  = [r.get("pb_f1",       0.0) or 0.0 for r in rows]
    pb_mf1 = [r.get("pb_macro_f1", 0.0) or 0.0 for r in rows]
    colors = [ALL_REP_COLORS.get(r["label"], "#999") for r in rows]
    n = len(rows)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ProcessBench (out-of-distribution transfer) performance",
                 fontsize=13, fontweight="bold")

    for ax, vals, title, ylim in [
        (axes[0], pb_f1,  "PB-F1 (first-error localization)", (0.0, 0.6)),
        (axes[1], pb_mf1, "PB Macro-F1",                      (0.3, 0.7)),
    ]:
        bars = ax.bar(x, vals, width=0.6, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(0.25, color="#aaa", lw=1, ls="--", label="25% threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Per-step LDA score line plots
# ---------------------------------------------------------------------------

def plot_step_lda(
    panels: list[tuple[str, dict | None]],
    out_path: Path,
    max_step: int = 10,
    min_count: int = 20,
) -> None:
    """For each representation, show mean LDA score ± 1 std per step position.

    Two lines: correct steps (blue) and incorrect steps (red). The gap between
    the lines reveals how much discriminative power the representation has at
    each step. If lines are parallel and flat, correctness signal is step-uniform;
    if they diverge or converge, step position matters.
    """
    loaded = []
    for key, data in panels:
        if data is None:
            print(f"  SKIP step-LDA: {key} (no data)")
            continue
        if data["step_pos"] is None:
            print(f"  SKIP step-LDA: {key} (no step_positions)")
            continue
        loaded.append((key, data))

    if not loaded:
        print("  No data for step-LDA figure.")
        return

    n = len(loaded)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Per-step LDA score: mean ± 1 std for correct vs incorrect steps\n"
        "(LDA trained on all steps; score normalised per representation to [0, 1])",
        fontsize=11, fontweight="bold",
    )

    for ax, (key, data) in zip(axes, loaded):
        h, y, step_pos = data["h"], data["y"], data["step_pos"]

        scores = _lda_scores(h, y)
        # Normalise to [0, 1] for cross-representation comparison
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)

        steps = sorted(s for s in set(step_pos.tolist()) if s <= max_step)

        mean_cor, std_cor, mean_inc, std_inc, xs = [], [], [], [], []
        for s in steps:
            mask = step_pos == s
            sc = scores[mask & (y == 1)]
            si = scores[mask & (y == 0)]
            if len(sc) < min_count or len(si) < min_count:
                continue
            xs.append(s)
            mean_cor.append(sc.mean()); std_cor.append(sc.std())
            mean_inc.append(si.mean()); std_inc.append(si.std())

        if not xs:
            ax.set_title(f"{KEY_LABELS.get(key, key)}\n(insufficient data per step)", fontsize=9)
            continue

        xs       = np.array(xs)
        mean_cor = np.array(mean_cor); std_cor = np.array(std_cor)
        mean_inc = np.array(mean_inc); std_inc = np.array(std_inc)

        ax.plot(xs, mean_cor, color=CORRECT_COL,   lw=2, marker="o", ms=5, label="correct")
        ax.fill_between(xs, mean_cor - std_cor, mean_cor + std_cor,
                        color=CORRECT_COL, alpha=FILL_ALPHA)

        ax.plot(xs, mean_inc, color=INCORRECT_COL, lw=2, marker="s", ms=5, ls="--",
                label="incorrect")
        ax.fill_between(xs, mean_inc - std_inc, mean_inc + std_inc,
                        color=INCORRECT_COL, alpha=FILL_ALPHA)

        # Annotate the gap at each step
        for xi, mc, mi in zip(xs, mean_cor, mean_inc):
            gap = mc - mi
            ax.annotate(f"{gap:+.2f}", xy=(xi, (mc + mi) / 2),
                        fontsize=6.5, ha="center", color="#555",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

        ax.set_xlabel("Step position", fontsize=9)
        ax.set_ylabel("LDA score (normalised)", fontsize=9)
        ax.set_title(KEY_LABELS.get(key, key), fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"s{int(s)}" for s in xs], fontsize=8)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: LDA score KDE per class (class separability)
# ---------------------------------------------------------------------------

def plot_lda_geometry(
    panels: list[tuple[str, dict | None]],
    out_path: Path,
) -> None:
    """KDE of LDA scores, one curve per class per representation.

    This is the most honest view of class separability: LDA finds the single
    best linear direction; if the classes overlap here, no linear probe can do
    better. Separation gap = mean(correct) - mean(incorrect) in LDA units.
    """
    loaded = [(k, d) for k, d in panels if d is not None]
    if not loaded:
        print("  No data for LDA geometry figure.")
        return

    n = len(loaded)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(
        "Class separability in the LDA direction\n"
        "(LDA = best linear axis for separating correct/incorrect; "
        "overlap = fundamental limit of linear probes)",
        fontsize=11, fontweight="bold",
    )

    for ax, (key, data) in zip(axes, loaded):
        h, y = data["h"], data["y"]
        scores = _lda_scores(h, y)

        sc = scores[y == 1]
        si = scores[y == 0]
        gap    = sc.mean() - si.mean()
        pooled = np.std(scores)
        cohen  = gap / (pooled + 1e-8)

        lo, hi = scores.min(), scores.max()
        xs = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 400)

        for vals, col, lbl in [(sc, CORRECT_COL, "correct"), (si, INCORRECT_COL, "incorrect")]:
            if vals.std() < 1e-8:
                continue
            kde = gaussian_kde(vals, bw_method="scott")
            ys  = kde(xs)
            ax.plot(xs, ys, color=col, lw=2, label=lbl)
            ax.fill_between(xs, ys, alpha=FILL_ALPHA, color=col)

        ax.axvline(sc.mean(), color=CORRECT_COL,   lw=1.2, ls=":")
        ax.axvline(si.mean(), color=INCORRECT_COL, lw=1.2, ls=":")

        ax.set_title(
            f"{KEY_LABELS.get(key, key)}\n"
            f"Cohen's d = {cohen:.3f}   gap = {gap:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("LDA score", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Per-solution LDA score trajectories
# ---------------------------------------------------------------------------

def _pick_solutions(
    sol_ids: np.ndarray,
    step_pos: np.ndarray,
    y: np.ndarray,
    min_steps: int = 3,
    n_correct: int = 8,
    n_incorrect: int = 8,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    unique = np.unique(sol_ids)
    cor, inc = [], []
    for sid in unique:
        mask = sol_ids == sid
        if mask.sum() < min_steps:
            continue
        labels = y[mask]
        if labels.all():
            cor.append(sid)
        elif not labels.any():
            inc.append(sid)
    rng.shuffle(cor); rng.shuffle(inc)
    return cor[:n_correct], inc[:n_incorrect]


def plot_lda_trajectories(
    panels: list[tuple[str, dict | None]],
    out_path: Path,
    n_correct: int = 8,
    n_incorrect: int = 8,
    min_steps: int = 3,
    max_step: int = 10,
) -> None:
    """Line plots of LDA score over steps for individual solutions.

    Each line = one solution; x = step position; y = LDA score.
    Blue solid = all-correct solutions, red dashed = all-incorrect solutions.
    A well-structured representation would show blue lines consistently above
    red, with convergent or monotone trajectories.
    """
    loaded = [
        (k, d) for k, d in panels
        if d is not None and d["sol_ids"] is not None and d["step_pos"] is not None
    ]
    if not loaded:
        print("  No data with solution_ids/step_positions for trajectory figure.")
        return

    n = len(loaded)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(
        "Per-solution LDA score across reasoning steps\n"
        "(each line = one solution; blue = all-correct, red = all-incorrect)",
        fontsize=11, fontweight="bold",
    )

    for ax, (key, data) in zip(axes, loaded):
        h, y, sol_ids, step_pos = data["h"], data["y"], data["sol_ids"], data["step_pos"]
        scores = _lda_scores(h, y)

        cor_sols, inc_sols = _pick_solutions(
            sol_ids, step_pos, y, min_steps, n_correct, n_incorrect)

        any_plotted = False
        for sol_list, col, ls, lbl in [
            (cor_sols, CORRECT_COL,   "-",  "correct"),
            (inc_sols, INCORRECT_COL, "--", "incorrect"),
        ]:
            first = True
            for sid in sol_list:
                mask  = sol_ids == sid
                steps = step_pos[mask]
                order = np.argsort(steps)
                s_seq = steps[order]
                v_seq = scores[mask][order]
                # Clip to max_step
                keep  = s_seq <= max_step
                s_seq = s_seq[keep]; v_seq = v_seq[keep]
                if len(s_seq) < 2:
                    continue
                ax.plot(s_seq, v_seq, color=col, lw=1.4, ls=ls, alpha=0.65,
                        label=lbl if first else None)
                ax.scatter(s_seq, v_seq, color=col, s=18, zorder=4, alpha=0.75)
                first = False
                any_plotted = True

        if not any_plotted:
            ax.text(0.5, 0.5, "no qualifying\nsolutions found",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#888")

        ax.set_title(KEY_LABELS.get(key, key), fontsize=9)
        ax.set_xlabel("Step position", fontsize=8)
        ax.set_ylabel("LDA score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise PTB experiment results")
    p.add_argument("--results",       required=True, help="summary_results.json")
    p.add_argument("--rep-dir",       required=True, help="Dir with *_ms_eval.npz files")
    p.add_argument("--dense-eval",    required=True, help="dense_eval_held_out.npz (backbone h_k)")
    p.add_argument("--ssae-eval",     default=None,  help="eval_held_out.npz (SSAE latents)")
    p.add_argument("--no-ssae-meta",  action="store_true",
                   help="Do not borrow step metadata from dense-eval for SSAE")
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--n-samples",     type=int, default=8000,
                   help="Max samples loaded per representation")
    p.add_argument("--n-traj",        type=int, default=8,
                   help="Solutions per class in trajectory plot")
    p.add_argument("--max-step",      type=int, default=10,
                   help="Max step position shown in step-LDA and trajectory plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rd   = Path(args.rep_dir)

    rows = json.loads(Path(args.results).read_text())["summary_rows"]

    # ---- Figs 1-2: performance tables ----
    print("Plotting Fig 1: MS performance ...")
    plot_ms_performance(rows, out / "fig1_ms_performance.png")

    print("Plotting Fig 2: PB transfer ...")
    plot_pb_transfer(rows, out / "fig2_pb_transfer.png")

    # ---- Load key representations ----
    # SSAE borrows step metadata from dense if available and row count matches
    ssae_meta = None if args.no_ssae_meta else args.dense_eval
    print("Loading key representation data ...")
    key_data: dict[str, dict | None] = {
        "dense":  _load(args.dense_eval,                 n_max=args.n_samples),
        "ssae_z": _load(args.ssae_eval,                  n_max=args.n_samples,
                        meta_src=ssae_meta) if args.ssae_eval else None,
        "ptb":    _load(rd / "ptb_no_l1_ms_eval.npz",    n_max=args.n_samples),
        "delta":  _load(rd / "dense_delta_ms_eval.npz",  n_max=args.n_samples),
    }
    panels = [(k, key_data[k]) for k in KEY_REPS]

    # ---- Fig 3: per-step LDA ----
    print("Plotting Fig 3: per-step LDA scores ...")
    plot_step_lda(panels, out / "fig3_step_lda.png", max_step=args.max_step)

    # ---- Fig 4: LDA geometry ----
    print("Plotting Fig 4: LDA score distributions ...")
    plot_lda_geometry(panels, out / "fig4_lda_geometry.png")

    # ---- Fig 5: LDA trajectories ----
    print("Plotting Fig 5: LDA trajectories ...")
    plot_lda_trajectories(
        panels, out / "fig5_lda_trajectories.png",
        n_correct=args.n_traj, n_incorrect=args.n_traj,
        max_step=args.max_step,
    )

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
