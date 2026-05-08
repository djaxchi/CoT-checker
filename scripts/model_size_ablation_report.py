#!/usr/bin/env python3
"""Model-size ablation: aggregate results across all Qwen2.5 sizes and generate report.

Loads metrics JSONs and score .npz files from model_size_ablation_probe.py (both
raw and l2 repr variants), plus the raw hidden-state .npz files from
model_size_ablation_extract.py.

--repr selects which variant to use for the per-model detail plots (histograms,
PCA, confusion matrices).  Both variants are shown together in the comparison
figure (08_raw_vs_l2.png).

Figures produced:
  01_probe_histograms.png     -- 4-panel probe-projection histograms (one per model)
  02_pca_scatter.png          -- 4-panel PCA plots (MS + PB projected into same space)
  03_auroc_vs_size.png        -- MS and PB AUROC vs model size
  04_macro_f1_vs_size.png     -- MS and PB Macro-F1 vs model size
  05_ppr_vs_size.png          -- Positive prediction rate vs model size
  06_confusion_matrices.png   -- 2x4 confusion matrix grid (MS top row, PB bottom)
  07_delta_vs_weights.png     -- Per-feature activation delta vs probe weight
  08_raw_vs_l2.png            -- Raw vs L2-norm AUROC comparison (both MS and PB)

Report written to: final_report.md

Usage:
    python scripts/model_size_ablation_report.py \\
        --data-dir   $SCRATCH/cot-checker/ms_ablation \\
        --scores-dir $SCRATCH/cot-checker/ms_ablation/probes \\
        --output-dir results/ms_ablation/ \\
        [--repr l2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.decomposition import PCA as SklearnPCA
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ---------------------------------------------------------------------------
# Model size ordering
# ---------------------------------------------------------------------------

TAGS        = ["0.5b", "1.5b", "3b", "7b"]
PARAM_COUNTS = [0.5,   1.5,    3.0,  7.0]   # in billions, for x-axis


# ---------------------------------------------------------------------------
# PCA (numpy, no sklearn dependency for the report)
# ---------------------------------------------------------------------------

def fit_pca(X: np.ndarray, n_components: int = 2):
    """Fit PCA on X.  Returns (components, mu) using sklearn if available, else numpy SVD."""
    mu = X.mean(axis=0)
    if _SKLEARN:
        pca = SklearnPCA(n_components=n_components, random_state=42)
        pca.fit(X - mu)
        return pca.components_, mu
    # Fallback: truncated numpy SVD (fine for small X)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:n_components], mu


def apply_pca(X: np.ndarray, components: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (X - mu) @ components.T


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

COLORS = {
    "ms_correct":   "#2196F3",
    "ms_incorrect": "#F44336",
    "pb_correct":   "#4CAF50",
    "pb_incorrect": "#FF9800",
}

SIZE_LABELS = {"0.5b": "0.5B", "1.5b": "1.5B", "3b": "3B", "7b": "7B"}


def _tag_title(tag: str, metrics: dict) -> str:
    ms = metrics["math_shepherd_eval"]
    pb = metrics["processbench"]
    return (
        f"Qwen2.5-{SIZE_LABELS[tag]}\n"
        f"MS F1={ms['macro_f1']:.3f}  AUROC={ms['auroc']:.3f}\n"
        f"PB F1={pb['macro_f1']:.3f}  AUROC={pb['auroc']:.3f}"
        + ("  [COLLAPSE]" if pb["collapse"] else "")
    )


# ---------------------------------------------------------------------------
# Plot 1: Probe-projection histograms
# ---------------------------------------------------------------------------

def plot_probe_histograms(all_scores: dict, all_metrics: dict, out: Path) -> None:
    available = [t for t in TAGS if t in all_scores]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax_i, tag in enumerate(available):
        ax     = axes[ax_i]
        sc     = all_scores[tag]
        ms_sc  = sc["ms_eval_scores"]
        ms_lab = sc["ms_eval_labels"]
        pb_sc  = sc["pb_scores"]
        pb_lab = sc["pb_labels"]
        thresh = all_metrics[tag]["best_threshold"]

        bins = np.linspace(0, 1, 51)
        ax.hist(ms_sc[ms_lab == 1], bins=bins, density=True, alpha=0.55,
                color=COLORS["ms_correct"],   label="MS correct")
        ax.hist(ms_sc[ms_lab == 0], bins=bins, density=True, alpha=0.55,
                color=COLORS["ms_incorrect"], label="MS incorrect")
        ax.hist(pb_sc[pb_lab == 1], bins=bins, density=True, alpha=0.40,
                color=COLORS["pb_correct"],   label="PB correct",   linestyle="--",
                histtype="step", linewidth=1.5)
        ax.hist(pb_sc[pb_lab == 0], bins=bins, density=True, alpha=0.40,
                color=COLORS["pb_incorrect"], label="PB incorrect", linestyle="--",
                histtype="step", linewidth=1.5)
        ax.axvline(thresh, color="black", linestyle=":", linewidth=1.2, label=f"thresh={thresh:.2f}")
        ax.set_title(_tag_title(tag, all_metrics[tag]), fontsize=8)
        ax.set_xlabel("P(correct)")
        ax.set_ylabel("Density")
        if ax_i == 0:
            ax.legend(fontsize=7, loc="upper left")

    for ax_i in range(len(available), 4):
        axes[ax_i].axis("off")

    fig.suptitle("Probe-projection histograms: MS (solid) vs PB (step)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: PCA scatter
# ---------------------------------------------------------------------------

def plot_pca_scatter(data_dir: Path, all_scores: dict, all_metrics: dict, out: Path) -> None:
    available = [t for t in TAGS if t in all_scores]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()
    MAX_PCA_FIT = 5_000   # samples used to fit PCA (controls SVD cost)

    for ax_i, tag in enumerate(available):
        ax = axes[ax_i]
        rng = np.random.default_rng(42)

        # Load hidden states
        ms_train_f = data_dir / f"ms_train_{tag}.npz"
        ms_eval_f  = data_dir / f"ms_eval_{tag}.npz"
        pb_f       = data_dir / f"pb_{tag}.npz"

        if not (ms_train_f.exists() and ms_eval_f.exists() and pb_f.exists()):
            ax.text(0.5, 0.5, "data missing", ha="center", va="center", transform=ax.transAxes)
            continue

        h_train = np.load(ms_train_f)["hidden_states"].astype(np.float32)
        h_eval  = np.load(ms_eval_f)["hidden_states"].astype(np.float32)
        y_eval  = np.load(ms_eval_f)["labels"].astype(np.int32)
        h_pb    = np.load(pb_f)["hidden_states"].astype(np.float32)
        y_pb    = np.load(pb_f)["step_labels"].astype(np.int32)

        # Fit PCA on MS train (subsampled for speed)
        n_pca = min(MAX_PCA_FIT, len(h_train))
        idx   = rng.choice(len(h_train), size=n_pca, replace=False)
        components, mu = fit_pca(h_train[idx], n_components=2)

        # Project eval + PB; subsample for plot clarity
        n_plot = min(3_000, len(h_eval))
        idx_e  = rng.choice(len(h_eval), size=n_plot, replace=False)
        Z_eval = apply_pca(h_eval[idx_e], components, mu)
        yy_e   = y_eval[idx_e]

        n_pb_plot = min(800, len(h_pb))
        idx_pb    = rng.choice(len(h_pb), size=n_pb_plot, replace=False)
        Z_pb      = apply_pca(h_pb[idx_pb], components, mu)
        yy_pb     = y_pb[idx_pb]

        # Probe direction projected onto PCA space
        coef = all_scores[tag]["probe_coef"][0].astype(np.float32)
        w_unit = coef / (np.linalg.norm(coef) + 1e-9)
        w_proj = components @ w_unit   # (2,)

        # Scatter: MS (circles), PB (triangles)
        for lab, mask in [(1, yy_e == 1), (0, yy_e == 0)]:
            c = COLORS["ms_correct"] if lab == 1 else COLORS["ms_incorrect"]
            ax.scatter(Z_eval[mask, 0], Z_eval[mask, 1], c=c, alpha=0.15, s=3,
                       marker="o", rasterized=True)
        for lab, mask in [(1, yy_pb == 1), (0, yy_pb == 0)]:
            c = COLORS["pb_correct"] if lab == 1 else COLORS["pb_incorrect"]
            ax.scatter(Z_pb[mask, 0], Z_pb[mask, 1], c=c, alpha=0.5, s=12,
                       marker="^", rasterized=True, edgecolors="none")

        # Probe direction arrow
        scale = np.percentile(np.abs(np.concatenate([Z_eval[:, :2], Z_pb[:, :2]])), 85)
        ax.annotate("", xy=(w_proj[0]*scale, w_proj[1]*scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        ax.set_title(_tag_title(tag, all_metrics[tag]), fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # Custom legend
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["ms_correct"],   markersize=6, label="MS correct"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["ms_incorrect"], markersize=6, label="MS incorrect"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor=COLORS["pb_correct"],   markersize=7, label="PB correct"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor=COLORS["pb_incorrect"], markersize=7, label="PB incorrect"),
        ]
        if ax_i == 0:
            ax.legend(handles=legend_handles, fontsize=7, loc="upper right")

    for ax_i in range(len(available), 4):
        axes[ax_i].axis("off")

    fig.suptitle(
        "PCA (fit on MS train) — MS eval (circles) + ProcessBench (triangles)\n"
        "Arrow = probe direction", fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3-5: Scaling plots
# ---------------------------------------------------------------------------

def plot_scaling_curves(all_metrics: dict, out_dir: Path) -> None:
    available = [t for t in TAGS if t in all_metrics]
    xs = [PARAM_COUNTS[TAGS.index(t)] for t in available]
    x_labels = [f"{x}B" for x in xs]

    def _get(dataset_key: str, metric: str) -> list[float]:
        return [all_metrics[t][dataset_key][metric] for t in available]

    # Figure 3: AUROC
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, _get("math_shepherd_eval", "auroc"), "o-", color="#2196F3", label="MS eval")
    ax.plot(xs, _get("processbench",       "auroc"), "s--", color="#FF9800", label="ProcessBench")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Model size (parameters)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax.legend()
    ax.set_title("AUROC vs model size")
    plt.tight_layout()
    fig.savefig(out_dir / "03_auroc_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '03_auroc_vs_size.png'}")

    # Figure 4: Macro-F1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, _get("math_shepherd_eval", "macro_f1"), "o-", color="#2196F3", label="MS eval")
    ax.plot(xs, _get("processbench",       "macro_f1"), "s--", color="#FF9800", label="ProcessBench")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Model size (parameters)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax.legend()
    ax.set_title("Macro-F1 vs model size")
    plt.tight_layout()
    fig.savefig(out_dir / "04_macro_f1_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '04_macro_f1_vs_size.png'}")

    # Figure 5: PPR
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, _get("math_shepherd_eval", "pos_pred_rate"), "o-", color="#2196F3", label="MS eval")
    ax.plot(xs, _get("processbench",       "pos_pred_rate"), "s--", color="#FF9800", label="ProcessBench")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Model size (parameters)")
    ax.set_ylabel("Positive prediction rate")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="0.5 reference")
    ax.legend()
    ax.set_title("Positive prediction rate (fraction predicted correct) vs model size")
    plt.tight_layout()
    fig.savefig(out_dir / "05_ppr_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '05_ppr_vs_size.png'}")


# ---------------------------------------------------------------------------
# Plot 6: Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(all_metrics: dict, out: Path) -> None:
    available = [t for t in TAGS if t in all_metrics]
    fig, axes = plt.subplots(2, len(available), figsize=(3.5 * len(available), 7))
    if len(available) == 1:
        axes = axes.reshape(2, 1)

    for col, tag in enumerate(available):
        for row, ds_key in enumerate(["math_shepherd_eval", "processbench"]):
            ax  = axes[row, col]
            cm  = np.array(all_metrics[tag][ds_key]["confusion_matrix"])  # [[TN FP], [FN TP]]
            im  = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred -", "Pred +"])
            ax.set_yticklabels(["True -", "True +"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                            color="white" if cm[i, j] > cm.max() * 0.5 else "black")
            ds_short = "MS eval" if "math" in ds_key else "PB"
            collapse_str = " [COLLAPSE]" if all_metrics[tag][ds_key]["collapse"] else ""
            ax.set_title(
                f"Qwen2.5-{SIZE_LABELS[tag]}\n{ds_short}  F1={all_metrics[tag][ds_key]['macro_f1']:.3f}{collapse_str}",
                fontsize=8,
            )
        plt.colorbar(im, ax=axes[:, col], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel("Math-Shepherd eval", fontsize=9)
    axes[1, 0].set_ylabel("ProcessBench GSM8K", fontsize=9)
    fig.suptitle("Confusion matrices — rows: true class, cols: predicted class", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 7: Delta vs weights (like mechanistic_analysis.py)
# ---------------------------------------------------------------------------

def plot_delta_vs_weights(data_dir: Path, all_scores: dict, out: Path) -> None:
    available = [t for t in TAGS if t in all_scores]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax_i, tag in enumerate(available):
        ax = axes[ax_i]
        ms_eval_f = data_dir / f"ms_eval_{tag}.npz"
        if not ms_eval_f.exists():
            ax.axis("off")
            continue

        h_eval = np.load(ms_eval_f)["hidden_states"].astype(np.float32)
        y_eval = np.load(ms_eval_f)["labels"].astype(np.int32)
        coef   = all_scores[tag]["probe_coef"][0].astype(np.float32)

        mean_cor = h_eval[y_eval == 1].mean(axis=0)
        mean_inc = h_eval[y_eval == 0].mean(axis=0)
        delta    = mean_cor - mean_inc

        # Subsample for scatter readability
        rng   = np.random.default_rng(42)
        n_sub = min(3000, len(coef))
        idx   = rng.choice(len(coef), size=n_sub, replace=False)

        ax.scatter(delta[idx], coef[idx], alpha=0.25, s=3, color="#555", rasterized=True)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        corr = float(np.corrcoef(delta, coef)[0, 1])
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                va="top", fontsize=10, fontweight="bold")
        ax.set_xlabel("mean_correct[i] - mean_incorrect[i]")
        ax.set_ylabel("Probe weight w[i]")
        ax.set_title(f"Qwen2.5-{SIZE_LABELS[tag]}  (dim={len(coef)})", fontsize=9)

    for ax_i in range(len(available), 4):
        axes[ax_i].axis("off")

    fig.suptitle("Per-feature: activation delta vs probe weight (r=1 means probe tracks mean diff)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(all_metrics: dict, out_path: Path) -> None:
    available = [t for t in TAGS if t in all_metrics]

    lines = [
        "# Model-Size Ablation: Dense Hidden-State Correctness Decodability",
        "",
        "Pilot experiment — 1 seed, dense_h only, last transformer layer last token.",
        f"Models evaluated: {', '.join('Qwen2.5-' + SIZE_LABELS[t] for t in available)}",
        "",
        "---",
        "",
        "## Per-Model Results",
        "",
        f"| Model | dim | MS AUROC | MS F1 | MS PPR | PB AUROC | PB F1 | PB PPR | Threshold | PB Collapse |",
        "|-------|-----|----------|-------|--------|----------|-------|--------|-----------|-------------|",
    ]
    for tag in available:
        m   = all_metrics[tag]
        ms  = m["math_shepherd_eval"]
        pb  = m["processbench"]
        col = "YES" if pb["collapse"] else "no"
        lines.append(
            f"| Qwen2.5-{SIZE_LABELS[tag]} | {m['hidden_dim']} "
            f"| {ms['auroc']:.4f} | {ms['macro_f1']:.4f} | {ms['pos_pred_rate']:.3f} "
            f"| {pb['auroc']:.4f} | {pb['macro_f1']:.4f} | {pb['pos_pred_rate']:.3f} "
            f"| {m['best_threshold']:.3f} | {col} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Research Questions",
        "",
    ]

    # Q1: Does MS decodability improve with size?
    ms_aurocs = [all_metrics[t]["math_shepherd_eval"]["auroc"] for t in available]
    ms_f1s    = [all_metrics[t]["math_shepherd_eval"]["macro_f1"] for t in available]
    ms_trend  = "increasing" if ms_aurocs[-1] > ms_aurocs[0] else "not consistently increasing"
    lines += [
        "### Q1: Does in-domain correctness decodability improve with model size?",
        "",
        f"MS AUROC values: {', '.join(f'{v:.4f}' for v in ms_aurocs)} (0.5B → 7B).",
        f"Trend: **{ms_trend}**.",
        "",
    ]

    # Q2: Does OOD PB transfer improve?
    pb_aurocs = [all_metrics[t]["processbench"]["auroc"] for t in available]
    pb_f1s    = [all_metrics[t]["processbench"]["macro_f1"] for t in available]
    pb_trend  = "improving" if pb_aurocs[-1] > pb_aurocs[0] else "not consistently improving"
    lines += [
        "### Q2: Does OOD ProcessBench transfer improve with model size?",
        "",
        f"PB AUROC values: {', '.join(f'{v:.4f}' for v in pb_aurocs)} (0.5B → 7B).",
        f"Trend: **{pb_trend}**.",
        "",
    ]

    # Q3: Geometric shift
    lines += [
        "### Q3: Is ProcessBench geometrically shifted relative to Math-Shepherd?",
        "",
        "See PCA scatter plots (02_pca_scatter.png). If PB triangles occupy a "
        "different region of the PCA plane than MS circles, a domain shift exists.",
        "",
    ]

    # Q4: Probe separability on PB
    pb_separable = [t for t in available if not all_metrics[t]["processbench"]["collapse"]
                    and all_metrics[t]["processbench"]["auroc"] > 0.55]
    lines += [
        "### Q4: Does the learned probe direction separate PB correct/incorrect?",
        "",
    ]
    if pb_separable:
        lines.append(
            f"PB AUROC > 0.55 for: {', '.join('Qwen2.5-' + SIZE_LABELS[t] for t in pb_separable)}. "
            "The probe transfers at least partially."
        )
    else:
        lines.append("PB AUROC <= 0.55 for all sizes. Probe direction does not transfer.")
    lines.append("")

    # Q5: Is 0.5B too small?
    if "0.5b" in all_metrics:
        ms05  = all_metrics["0.5b"]["math_shepherd_eval"]["auroc"]
        pb05  = all_metrics["0.5b"]["processbench"]["auroc"]
        issue = "0.5B MS AUROC is low" if ms05 < 0.55 else "0.5B has reasonable in-domain AUROC"
        lines += [
            "### Q5: Is Qwen2.5-0.5B likely too small, or is the main issue dataset transfer?",
            "",
            f"0.5B MS AUROC={ms05:.4f}, PB AUROC={pb05:.4f}. {issue}. "
            "Compare: if all models have low PB AUROC regardless of MS AUROC, the bottleneck is "
            "distribution shift, not model capacity.",
            "",
        ]

    # Q6: Recommended model for PTB/SSAE
    best_joint_tag = max(
        available,
        key=lambda t: (
            all_metrics[t]["math_shepherd_eval"]["auroc"] +
            all_metrics[t]["processbench"]["auroc"]
        ),
    )
    lines += [
        "### Q6: Which model size is worth using for PTB/SSAE follow-up?",
        "",
        f"Best joint (MS + PB) AUROC: **Qwen2.5-{SIZE_LABELS[best_joint_tag]}**.",
        "Use this size for the full multi-seed PTB/SSAE experiment unless resource constraints apply.",
        "",
        "---",
        "",
        "## Figures",
        "",
        "- `01_probe_histograms.png`  -- Probe score distributions (MS solid, PB step)",
        "- `02_pca_scatter.png`       -- PCA: MS eval (circles) + PB (triangles), arrow = probe dir",
        "- `03_auroc_vs_size.png`     -- AUROC scaling curve",
        "- `04_macro_f1_vs_size.png`  -- Macro-F1 scaling curve",
        "- `05_ppr_vs_size.png`       -- Positive prediction rate scaling curve",
        "- `06_confusion_matrices.png`-- Confusion matrices (MS top, PB bottom)",
        "- `07_delta_vs_weights.png`  -- Per-feature activation delta vs probe weight",
    ]

    report_text = "\n".join(lines)
    out_path.write_text(report_text)
    print(f"\n  Saved report → {out_path}")
    print()
    print(report_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model-size ablation: aggregate report and plots")
    p.add_argument("--data-dir",   required=True,
                   help="Directory with ms_train/eval and pb .npz files from extraction")
    p.add_argument("--scores-dir", required=True,
                   help="Directory with metrics_*.json and scores_*.npz from probe script")
    p.add_argument("--output-dir", default="results/ms_ablation/",
                   help="Directory to write figures and final_report.md")
    p.add_argument("--repr",       default="l2", choices=["raw", "l2"],
                   help="Which repr variant to use for per-model detail plots.  "
                        "Both variants are always shown in the raw-vs-l2 comparison figure.")
    return p.parse_args()


def plot_raw_vs_l2(all_metrics_by_repr: dict[str, dict], out: Path) -> None:
    """Figure 08: AUROC for raw vs l2 variants, side by side (MS and PB)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    markers = {"raw": "o", "l2": "s"}
    styles  = {"raw": "-", "l2": "--"}
    colors  = {"raw": "#E91E63", "l2": "#2196F3"}

    for ax, ds_key, ds_label in [
        (axes[0], "math_shepherd_eval", "Math-Shepherd eval"),
        (axes[1], "processbench",       "ProcessBench GSM8K"),
    ]:
        for repr_mode, all_metrics in all_metrics_by_repr.items():
            available = [t for t in TAGS if t in all_metrics]
            if not available:
                continue
            xs   = [PARAM_COUNTS[TAGS.index(t)] for t in available]
            aurocs = [all_metrics[t][ds_key]["auroc"] for t in available]
            ax.plot(xs, aurocs,
                    marker=markers[repr_mode], linestyle=styles[repr_mode],
                    color=colors[repr_mode], label=repr_mode)

        ax.set_xticks([PARAM_COUNTS[TAGS.index(t)] for t in TAGS])
        ax.set_xticklabels([f"{x}B" for x in PARAM_COUNTS])
        ax.set_xlabel("Model size (parameters)")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.45, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
        ax.set_title(f"AUROC — {ds_label}")
        ax.legend()

    fig.suptitle("Raw hidden states vs L2-normalised: does magnitude carry signal?",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    args = parse_args()
    data_dir   = Path(args.data_dir)
    scores_dir = Path(args.scores_dir)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_repr = args.repr

    # --- Load metrics + scores for each available repr variant ---
    all_metrics_by_repr: dict[str, dict] = {"raw": {}, "l2": {}}
    all_scores_by_repr:  dict[str, dict] = {"raw": {}, "l2": {}}

    for repr_mode in ("raw", "l2"):
        for tag in TAGS:
            suffix = f"{tag}_{repr_mode}"
            m_path = scores_dir / f"metrics_{suffix}.json"
            s_path = scores_dir / f"scores_{suffix}.npz"
            if not m_path.exists() or not s_path.exists():
                continue
            with open(m_path) as f:
                all_metrics_by_repr[repr_mode][tag] = json.load(f)
            sd = np.load(s_path)
            all_scores_by_repr[repr_mode][tag] = {k: sd[k] for k in sd.files}
            ms_f1 = all_metrics_by_repr[repr_mode][tag]["math_shepherd_eval"]["macro_f1"]
            pb_f1 = all_metrics_by_repr[repr_mode][tag]["processbench"]["macro_f1"]
            print(f"  Loaded {repr_mode}/{tag}: MS F1={ms_f1:.4f}  PB F1={pb_f1:.4f}")

    # Select primary repr for detail plots
    all_metrics = all_metrics_by_repr[primary_repr]
    all_scores  = all_scores_by_repr[primary_repr]

    if not all_metrics:
        print(f"ERROR: no metrics found for repr='{primary_repr}'. "
              f"Run model_size_ablation_probe.py --repr {primary_repr} first.")
        sys.exit(1)

    # --- Generate figures ---
    print(f"\nGenerating figures (primary repr={primary_repr}) …")

    plot_probe_histograms(all_scores, all_metrics, out_dir / "01_probe_histograms.png")

    plot_pca_scatter(data_dir, all_scores, all_metrics, out_dir / "02_pca_scatter.png")

    plot_scaling_curves(all_metrics, out_dir)

    plot_confusion_matrices(all_metrics, out_dir / "06_confusion_matrices.png")

    plot_delta_vs_weights(data_dir, all_scores, out_dir / "07_delta_vs_weights.png")

    # Raw vs L2 comparison (uses both repr variants if available)
    present_reprs = {r: m for r, m in all_metrics_by_repr.items() if m}
    if len(present_reprs) >= 2:
        plot_raw_vs_l2(present_reprs, out_dir / "08_raw_vs_l2.png")
    elif len(present_reprs) == 1:
        print(f"  Skipping 08_raw_vs_l2.png (only one repr variant available: "
              f"{list(present_reprs.keys())[0]})")

    # --- Write text report ---
    write_report(all_metrics, out_dir / "final_report.md")

    print(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    main()
