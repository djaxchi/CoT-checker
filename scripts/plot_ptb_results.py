#!/usr/bin/env python3
"""Visualise PTB experiment results.

Produces five figures:
  fig1_ms_performance.png       -- Math-Shepherd metrics (in-distribution) for all representations
  fig2_pb_transfer.png          -- ProcessBench transfer metrics (separate from MS)
  fig3_activation_heatmap.png   -- Differential activation heatmap: top features x step position,
                                   colour = mean(correct) - mean(incorrect), for 4 key representations
  fig4_geometry.png             -- 2D PCA scatter coloured by correctness
  fig5_trajectories.png         -- Step-by-step solution paths through PCA space

The four representations highlighted in figs 3-5:
  dense   = raw backbone h_k           (--dense-eval)
  ssae_z  = SSAE sparse latents        (--ssae-eval, optional)
  ptb     = PTB bottleneck (no_l1)     (--rep-dir / ptb_no_l1_ms_eval.npz)
  delta   = dense transition Δh        (--rep-dir / dense_delta_ms_eval.npz)

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
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORRECT_COL   = "#2196F3"
INCORRECT_COL = "#F44336"
ALPHA_BG      = 0.20

# Order and display names for the full benchmark table (figs 1-2)
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

# The four key representations for detailed figures (3-5)
KEY_REPS = ["dense", "ssae_z", "ptb", "delta"]
KEY_LABELS = {
    "dense":  "Dense h_k\n(backbone)",
    "ssae_z": "SSAE z_k\n(sparse features)",
    "ptb":    "PTB\n(transition bottleneck)",
    "delta":  "Dense Δh\n(transition delta)",
}
KEY_COLORS = {
    "dense":  "#546E7A",
    "ssae_z": "#7B1FA2",
    "ptb":    "#2E7D32",
    "delta":  "#1565C0",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load(path: str | Path, n_max: int | None = None, seed: int = 0) -> dict | None:
    p = Path(path)
    if not p.exists():
        print(f"  MISSING: {p}")
        return None
    d = np.load(p)
    h = d["latents"].astype(np.float32)
    y_key = "correctness" if "correctness" in d.files else ("step_labels" if "step_labels" in d.files else None)
    if y_key is None:
        print(f"  WARNING: no correctness key in {p.name}")
        return None
    y = d[y_key].astype(int)
    sol_ids  = d["solution_ids"].astype(int)  if "solution_ids"  in d.files else None
    step_pos = d["step_positions"].astype(int) if "step_positions" in d.files else None

    if n_max is not None and len(h) > n_max:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(h), n_max, replace=False)
        idx.sort()
        h = h[idx]; y = y[idx]
        if sol_ids  is not None: sol_ids  = sol_ids[idx]
        if step_pos is not None: step_pos = step_pos[idx]

    return {"h": h, "y": y, "sol_ids": sol_ids, "step_pos": step_pos}


# ---------------------------------------------------------------------------
# Figure 1: MS Performance (Math-Shepherd, in-distribution)
# ---------------------------------------------------------------------------

def plot_ms_performance(summary_rows: list[dict], out_path: Path) -> None:
    rows = [r for r in summary_rows if r["label"] in ALL_REP_ORDER]
    rows.sort(key=lambda r: ALL_REP_ORDER.index(r["label"]))

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
        (axes[0], ms_mf1,   ms_std,  "Macro-F1 (balanced threshold)", (0.60, 0.80)),
        (axes[1], ms_auroc, [0]*n,   "AUROC",                         (0.70, 0.90)),
    ]:
        bars = ax.bar(x, vals, width=0.6, color=colors, edgecolor="white", linewidth=0.8,
                      yerr=stds, capsize=3, error_kw={"ecolor": "#444", "lw": 1.2})
        ax.axhline(0.5, color="#aaa", lw=1, ls="--", label="chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (stds[list(vals).index(val)] if stds else 0) + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: PB Transfer (ProcessBench, out-of-distribution)
# ---------------------------------------------------------------------------

def plot_pb_transfer(summary_rows: list[dict], out_path: Path) -> None:
    rows = [r for r in summary_rows if r["label"] in ALL_REP_ORDER]
    rows.sort(key=lambda r: ALL_REP_ORDER.index(r["label"]))

    labels  = [ALL_REP_LABELS.get(r["label"], r["label"]) for r in rows]
    pb_f1   = [r.get("pb_f1",       0.0) or 0.0 for r in rows]
    pb_mf1  = [r.get("pb_macro_f1", 0.0) or 0.0 for r in rows]
    colors  = [ALL_REP_COLORS.get(r["label"], "#999") for r in rows]

    n = len(rows)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ProcessBench (out-of-distribution transfer) performance", fontsize=13, fontweight="bold")

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
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Differential activation heatmap
# ---------------------------------------------------------------------------

def _compute_step_activations(
    data: dict,
    top_k: int = 50,
    max_step: int = 10,
) -> dict | None:
    """Compute per-step mean activations for correct vs incorrect samples.

    Returns dict with:
        steps       : sorted unique step positions (clipped to max_step)
        feat_idx    : top_k feature indices ranked by discriminative power
        mean_correct: (top_k, n_steps) mean activation for correct steps
        mean_incorr : (top_k, n_steps) mean activation for incorrect steps
        diff        : mean_correct - mean_incorr  (same shape)
    """
    h, y, step_pos = data["h"], data["y"], data["step_pos"]
    if step_pos is None:
        return None

    steps = sorted(set(step_pos.tolist()))
    steps = [s for s in steps if s <= max_step]
    if not steps:
        return None

    n_steps = len(steps)
    D = h.shape[1]

    mean_cor = np.zeros((D, n_steps), dtype=np.float32)
    mean_inc = np.zeros((D, n_steps), dtype=np.float32)

    for j, s in enumerate(steps):
        mask = step_pos == s
        h_s = h[mask]
        y_s = y[mask]
        cor = y_s == 1
        inc = y_s == 0
        if cor.any():
            mean_cor[:, j] = h_s[cor].mean(axis=0)
        if inc.any():
            mean_inc[:, j] = h_s[inc].mean(axis=0)

    # Select top_k most discriminative features by sum of |diff| across steps
    diff = mean_cor - mean_inc
    disc = np.abs(diff).sum(axis=1)
    feat_idx = np.argsort(disc)[::-1][:top_k]

    # Sort selected features by the step where they peak (for visual clarity)
    peak_steps = np.argmax(np.abs(mean_cor[feat_idx] - mean_inc[feat_idx]), axis=1)
    sort_order = np.argsort(peak_steps)
    feat_idx = feat_idx[sort_order]

    return {
        "steps":        steps,
        "feat_idx":     feat_idx,
        "mean_correct": mean_cor[feat_idx],
        "mean_incorr":  mean_inc[feat_idx],
        "diff":         diff[feat_idx],
    }


def plot_activation_heatmap(
    panels: list[tuple[str, dict | None]],   # (key, data_dict)
    out_path: Path,
    top_k: int = 50,
    max_step: int = 10,
) -> None:
    """Differential activation heatmap: colour = mean_correct - mean_incorrect.

    Each panel shows top_k most discriminative features (y-axis) across step
    positions (x-axis). Blue = more active on correct steps; red = more active
    on incorrect steps; white = no difference.
    """
    computed = []
    for key, data in panels:
        if data is None:
            print(f"  SKIP heatmap panel: {key} (no data)")
            continue
        result = _compute_step_activations(data, top_k=top_k, max_step=max_step)
        if result is None:
            print(f"  SKIP heatmap panel: {key} (no step_positions)")
            continue
        computed.append((key, result))

    if not computed:
        print("  No data for activation heatmap.")
        return

    n_panels = len(computed)
    # Layout: each representation gets a row, with 3 sub-columns (correct, incorrect, diff)
    fig, axes = plt.subplots(n_panels, 3, figsize=(15, 4 * n_panels),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.45})
    if n_panels == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Activation profiles: top discriminative features across reasoning steps\n"
        "(features ranked by |correct − incorrect| summed over steps)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    vmax_diff = 0.0
    for _, res in computed:
        vmax_diff = max(vmax_diff, np.abs(res["diff"]).max())
    vmax_diff = vmax_diff or 1.0

    # Per-panel colour limits (normalise correct/incorrect independently)
    cmap_act  = "Blues"
    cmap_diff = "RdBu_r"

    col_titles = ["Mean activation\n(correct steps)", "Mean activation\n(incorrect steps)",
                  "Differential\n(correct − incorrect)"]

    for row_i, (key, res) in enumerate(computed):
        steps       = res["steps"]
        diff        = res["diff"]
        mean_cor    = res["mean_correct"]
        mean_inc    = res["mean_incorr"]

        step_labels = [f"s{s}" for s in steps]
        feat_labels = [str(i) for i in res["feat_idx"]]
        show_yticks = top_k <= 20

        vmax_act = max(mean_cor.max(), mean_inc.max(), 1e-6)

        for col_i, (mat, cmap, vmin, vmax) in enumerate([
            (mean_cor, cmap_act,  0,           vmax_act),
            (mean_inc, cmap_act,  0,           vmax_act),
            (diff,     cmap_diff, -vmax_diff,  vmax_diff),
        ]):
            ax = axes[row_i, col_i]
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels(step_labels, fontsize=8)
            ax.set_xlabel("Step position", fontsize=8)

            if show_yticks:
                ax.set_yticks(range(top_k))
                ax.set_yticklabels(feat_labels, fontsize=6)
                ax.set_ylabel("Feature index", fontsize=8)
            else:
                ax.set_yticks([])
                ax.set_ylabel(f"Top {top_k} features\n(ranked by discriminability)", fontsize=8)

            title = col_titles[col_i]
            if col_i == 0:
                title = f"{KEY_LABELS.get(key, key)}\n\n{title}"
            ax.set_title(title, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Geometry (PCA scatter)
# ---------------------------------------------------------------------------

def plot_geometry(
    panels: list[tuple[str, dict | None]],
    out_path: Path,
) -> None:
    loaded = [(k, d) for k, d in panels if d is not None]
    if not loaded:
        print("  No data for geometry figure.")
        return

    n_panels = len(loaded)
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle("Representation geometry (PCA, coloured by step correctness)",
                 fontsize=12, fontweight="bold")

    for ax, (key, data) in zip(axes, loaded):
        h, y = data["h"], data["y"]
        pca = PCA(n_components=2, random_state=0)
        z2  = pca.fit_transform(h)
        ev  = pca.explained_variance_ratio_

        ax.scatter(z2[y==0, 0], z2[y==0, 1], c=INCORRECT_COL, alpha=ALPHA_BG,
                   s=4, linewidths=0, rasterized=True, label="incorrect")
        ax.scatter(z2[y==1, 0], z2[y==1, 1], c=CORRECT_COL,   alpha=ALPHA_BG,
                   s=4, linewidths=0, rasterized=True, label="correct")

        ax.set_title(f"{KEY_LABELS.get(key, key)}\n(PC1={ev[0]:.1%}, PC2={ev[1]:.1%})", fontsize=10)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_els = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=CORRECT_COL,   markersize=8, label="correct"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=INCORRECT_COL, markersize=8, label="incorrect"),
    ]
    axes[min(n_panels-1, len(axes)-1)].legend(handles=legend_els, loc="best", fontsize=9)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Step trajectories
# ---------------------------------------------------------------------------

def _pick_solutions(
    sol_ids: np.ndarray,
    step_pos: np.ndarray,
    y: np.ndarray,
    min_steps: int = 4,
    n_correct: int = 5,
    n_incorrect: int = 5,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    unique_sols = np.unique(sol_ids)
    correct_sols, incorrect_sols = [], []
    for sid in unique_sols:
        mask   = sol_ids == sid
        steps  = step_pos[mask]
        if len(steps) < min_steps:
            continue
        labels = y[mask]
        if labels.all():
            correct_sols.append(sid)
        elif not labels.any():
            incorrect_sols.append(sid)
    rng.shuffle(correct_sols)
    rng.shuffle(incorrect_sols)
    return correct_sols[:n_correct], incorrect_sols[:n_incorrect]


def plot_trajectories(
    panels: list[tuple[str, dict | None]],
    out_path: Path,
    n_bg: int = 3000,
    n_correct: int = 5,
    n_incorrect: int = 5,
    min_steps: int = 4,
) -> None:
    loaded = [(k, d) for k, d in panels
              if d is not None and d["sol_ids"] is not None and d["step_pos"] is not None]
    if not loaded:
        print("  No data with solution_ids/step_positions for trajectory figure.")
        return

    n_panels = len(loaded)
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle("Step-by-step trajectories in representation space (PCA)",
                 fontsize=12, fontweight="bold")

    for ax, (key, data) in zip(axes, loaded):
        h, y, sol_ids, step_pos = data["h"], data["y"], data["sol_ids"], data["step_pos"]

        rng_np = np.random.default_rng(0)
        bg_idx = rng_np.choice(len(h), min(n_bg, len(h)), replace=False)
        pca    = PCA(n_components=2, random_state=0)
        pca.fit(h[bg_idx])
        ev = pca.explained_variance_ratio_

        z_bg = pca.transform(h[bg_idx])
        y_bg = y[bg_idx]
        ax.scatter(z_bg[y_bg==0, 0], z_bg[y_bg==0, 1], c=INCORRECT_COL, alpha=0.07,
                   s=3, linewidths=0, rasterized=True)
        ax.scatter(z_bg[y_bg==1, 0], z_bg[y_bg==1, 1], c=CORRECT_COL,   alpha=0.07,
                   s=3, linewidths=0, rasterized=True)

        cor_sols, inc_sols = _pick_solutions(
            sol_ids, step_pos, y, min_steps, n_correct, n_incorrect)

        for sol_list, linestyle in [(cor_sols, "-"), (inc_sols, "--")]:
            for sid in sol_list:
                mask  = sol_ids == sid
                steps = step_pos[mask]
                order = np.argsort(steps)
                h_sol = h[mask][order]
                y_sol = y[mask][order]
                z_sol = pca.transform(h_sol)

                for k in range(len(z_sol) - 1):
                    seg_col = CORRECT_COL if y_sol[k] == 1 else INCORRECT_COL
                    ax.annotate(
                        "", xy=(z_sol[k+1, 0], z_sol[k+1, 1]),
                        xytext=(z_sol[k, 0], z_sol[k, 1]),
                        arrowprops=dict(arrowstyle="-|>", color=seg_col,
                                        lw=1.5, mutation_scale=10),
                    )
                for k, (pt, lbl) in enumerate(zip(z_sol, y_sol)):
                    c = CORRECT_COL if lbl == 1 else INCORRECT_COL
                    ax.scatter(*pt, c=c, s=40, zorder=5, edgecolors="white", linewidths=0.6)
                    ax.text(pt[0], pt[1], str(k+1), fontsize=6, ha="center", va="center",
                            color="white", fontweight="bold", zorder=6)

        ax.set_title(f"{KEY_LABELS.get(key, key)}\n(PC1={ev[0]:.1%}, PC2={ev[1]:.1%})", fontsize=10)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_els = [
        Line2D([0],[0], color=CORRECT_COL,   lw=2,        label="all-correct solution"),
        Line2D([0],[0], color=INCORRECT_COL, lw=2, ls="--", label="all-incorrect solution"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#aaa", markersize=8,
               label="step (number = position)"),
    ]
    axes[min(n_panels-1, len(axes)-1)].legend(handles=legend_els, loc="best", fontsize=8)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise PTB experiment results")
    p.add_argument("--results",    required=True, help="summary_results.json from eval script")
    p.add_argument("--rep-dir",    required=True, help="Dir with *_ms_eval.npz files (PTB variants + baselines)")
    p.add_argument("--dense-eval", required=True, help="dense_eval_held_out.npz (raw backbone h_k)")
    p.add_argument("--ssae-eval",  default=None,  help="eval_held_out.npz (SSAE sparse latents, optional)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-bg",       type=int, default=5000, help="Max samples for geometry/trajectory plots")
    p.add_argument("--n-traj",     type=int, default=5,    help="Solutions per class in trajectory plot")
    p.add_argument("--top-k",      type=int, default=50,   help="Top K features in activation heatmap")
    p.add_argument("--max-step",   type=int, default=10,   help="Max step position in heatmap")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rd   = Path(args.rep_dir)

    data = json.loads(Path(args.results).read_text())
    rows = data["summary_rows"]

    # ---- Figures 1-2: Performance tables (all representations) ----
    print("Plotting Fig 1: MS performance ...")
    plot_ms_performance(rows, out / "fig1_ms_performance.png")

    print("Plotting Fig 2: PB transfer ...")
    plot_pb_transfer(rows, out / "fig2_pb_transfer.png")

    # ---- Load the four key representation datasets ----
    print("Loading key representation data ...")
    key_data: dict[str, dict | None] = {
        "dense":  _load(args.dense_eval,             n_max=args.n_bg),
        "ssae_z": _load(args.ssae_eval,              n_max=args.n_bg) if args.ssae_eval else None,
        "ptb":    _load(rd / "ptb_no_l1_ms_eval.npz", n_max=args.n_bg),
        "delta":  _load(rd / "dense_delta_ms_eval.npz", n_max=args.n_bg),
    }

    key_panels = [(k, key_data[k]) for k in KEY_REPS]

    # ---- Figure 3: Activation heatmap ----
    print("Plotting Fig 3: activation heatmap ...")
    plot_activation_heatmap(
        key_panels, out / "fig3_activation_heatmap.png",
        top_k=args.top_k, max_step=args.max_step,
    )

    # ---- Figure 4: Geometry ----
    print("Plotting Fig 4: geometry ...")
    plot_geometry(key_panels, out / "fig4_geometry.png")

    # ---- Figure 5: Trajectories ----
    print("Plotting Fig 5: trajectories ...")
    plot_trajectories(
        key_panels, out / "fig5_trajectories.png",
        n_bg=args.n_bg, n_correct=args.n_traj, n_incorrect=args.n_traj,
    )

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
