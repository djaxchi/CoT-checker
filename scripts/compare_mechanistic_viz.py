#!/usr/bin/env python3
"""Compare mechanistic properties of N encoding strategies side by side.

Generates multi-panel comparison figures (4 rows × N columns):
  Row 1: PCA scatter (PC1 vs PC2) -- correct=blue, incorrect=red, probe direction arrow
  Row 2: 1D projection onto probe axis (density histograms)
  Row 3: Activation delta vs probe weight (r= annotated)
  Row 4: Active dimensions histogram

One figure per dataset (Math-Shepherd eval subset, ProcessBench).

Runs CPU-only from pre-encoded .npz files and saved linear probe checkpoints.

Config JSON format:
  {
    "encodings": [
      {
        "label":        "Dense h_k",
        "ms_npz":       "/path/dense_eval_held_out.npz",
        "pb_npz":       "/path/processbench_dense_gsm8k.npz",
        "probes":       ["/path/dense_linear_probe_seed42.pt", ...],
        "ms_label_key": "correctness",
        "pb_label_key": "step_labels"
      },
      ...
    ],
    "n_samples_ms":  5000,
    "n_samples_pb":  0,
    "reducer":       "pca"
  }

Usage:
  python scripts/compare_mechanistic_viz.py \\
      --config results/mechanistic_comparison/config.json \\
      --output-dir results/mechanistic_comparison/
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

BLUE = "#1565C0"
RED  = "#C62828"
GREY = "#424242"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_encoding(npz_path: str, label_key: str, n_samples: int, seed: int = 42
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    """Load latents + labels from npz, optionally balance-subsample.

    n_samples=0 means use all available (balanced by the minority class).
    Returns None if file not found.
    """
    p = Path(npz_path)
    if not p.exists():
        print(f"  WARNING: {npz_path} not found -- skipping this encoding for this dataset")
        return None
    d = np.load(p)
    h = d["latents"].astype(np.float32)
    y = d[label_key].astype(np.int32)

    # Balance by the minority class
    rng = np.random.default_rng(seed)
    cor_idx = np.where(y == 1)[0]
    inc_idx = np.where(y == 0)[0]
    n_min = min(len(cor_idx), len(inc_idx))
    if n_samples > 0:
        n_min = min(n_min, n_samples // 2)
    sel = np.concatenate([
        rng.choice(cor_idx, n_min, replace=False),
        rng.choice(inc_idx, n_min, replace=False),
    ])
    rng.shuffle(sel)
    print(f"  {p.name}: {len(sel):,} steps (balanced)  dim={h.shape[1]}")
    return h[sel], y[sel]


def load_probe_weights(probe_paths: list[str]) -> tuple[np.ndarray, float]:
    """Average linear probe weights across seeds. Returns (w, b)."""
    ws, bs = [], []
    for path in probe_paths:
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: probe {path} not found -- skipping")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        sd = ckpt["state_dict"]
        ws.append(sd["fc.weight"].numpy().squeeze())
        bs.append(sd["fc.bias"].numpy().squeeze())
    if not ws:
        return None, None
    return np.stack(ws).mean(0), float(np.array(bs).mean())


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def numpy_pca(X: np.ndarray, n: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated PCA via SVD. Returns (Z, components, explained_variance_ratio)."""
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(len(X) - 1, 1)
    expl = var[:n] / var.sum()
    Z = Xc @ Vt[:n].T
    return Z, Vt[:n], expl


def try_umap(X: np.ndarray, seed: int = 42) -> np.ndarray | None:
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return reducer.fit_transform(X)
    except ImportError:
        return None


def compute_stats(h: np.ndarray, y: np.ndarray, w: np.ndarray
                  ) -> dict:
    """Compute all per-encoding statistics needed for the figure."""
    w_unit = w / (np.linalg.norm(w) + 1e-9)
    cor_mask = y == 1
    inc_mask = y == 0

    mean_cor = h[cor_mask].mean(0)
    mean_inc = h[inc_mask].mean(0)
    delta    = mean_cor - mean_inc
    r_corr   = float(np.corrcoef(w, delta)[0, 1])

    proj_1d = h @ w_unit
    active  = (h > 0).sum(1)

    return {
        "delta":   delta,
        "r_corr":  r_corr,
        "proj_1d": proj_1d,
        "active":  active,
        "w_unit":  w_unit,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _row_pca(ax, h: np.ndarray, y: np.ndarray, w_unit: np.ndarray,
             r_corr: float, label: str, reducer: str = "pca") -> None:
    Z = None
    if reducer == "umap":
        Z = try_umap(h)
    if Z is None:
        Z, comps, expl = numpy_pca(h)
        w_proj = comps @ w_unit
        xlabel = f"PC1 ({expl[0]*100:.1f}%)"
        ylabel = f"PC2 ({expl[1]*100:.1f}%)"
    else:
        w_proj = np.array([0.0, 0.0])  # no meaningful direction for UMAP
        xlabel, ylabel = "UMAP-1", "UMAP-2"

    for cls, color, name in [(1, BLUE, "correct"), (0, RED, "incorrect")]:
        mask = y == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], c=color, alpha=0.2, s=4,
                   label=name, rasterized=True)

    if reducer == "pca" and (np.abs(w_proj[:2]) > 1e-6).any():
        scale = np.percentile(np.abs(Z), 85)
        dx, dy = w_proj[0] * scale, w_proj[1] * scale
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2.0))

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(f"{label}\nr={r_corr:.3f}", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)


def _row_hist1d(ax, proj_1d: np.ndarray, y: np.ndarray) -> None:
    bins = np.linspace(proj_1d.min(), proj_1d.max(), 60)
    ax.hist(proj_1d[y == 1], bins=bins, density=True, alpha=0.65,
            color=BLUE, label="correct")
    ax.hist(proj_1d[y == 0], bins=bins, density=True, alpha=0.65,
            color=RED, label="incorrect")
    ax.set_xlabel("w·h (probe score)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)


def _row_delta_weights(ax, delta: np.ndarray, w: np.ndarray, r_corr: float) -> None:
    ax.scatter(delta, w, alpha=0.25, s=2, color=GREY, rasterized=True)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("activation delta (cor−inc)", fontsize=8)
    ax.set_ylabel("probe weight w[i]", fontsize=8)
    ax.text(0.05, 0.95, f"r = {r_corr:.3f}", transform=ax.transAxes,
            va="top", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)


def _row_active_dims(ax, active: np.ndarray, y: np.ndarray, D: int) -> None:
    counts = np.unique(active)
    if len(counts) <= 2:
        # Deterministic (TopK or dense): just annotate
        ax.axvline(counts[0], color=BLUE, lw=2, label=f"all = {counts[0]}")
        ax.set_xlim(max(0, counts[0] - 5), min(D, counts[0] + 5))
        ax.set_title(f"Active dims = {counts[0]}/{D}", fontsize=8)
    else:
        lo, hi = int(active.min()), int(active.max())
        bins = np.arange(lo, hi + 2)
        ax.hist(active[y == 1], bins=bins, density=True, alpha=0.65,
                color=BLUE, label="correct")
        ax.hist(active[y == 0], bins=bins, density=True, alpha=0.65,
                color=RED, label="incorrect")
    ax.set_xlabel(f"# active dims  (D={D})", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)


def make_comparison_figure(
    enc_data: list[dict],
    dataset_title: str,
    reducer: str = "pca",
) -> plt.Figure:
    """Build 4-row × N-col comparison figure.

    enc_data: list of dicts with keys:
      label, h, y, w, stats
    """
    N = len(enc_data)
    fig = plt.figure(figsize=(5 * N, 17))
    fig.suptitle(dataset_title, fontsize=13, fontweight="bold", y=0.995)

    row_labels = ["PCA scatter", "1D probe projection",
                  "Δactivation vs probe weight", "Active dimensions"]
    row_heights = [4.5, 3.0, 3.5, 3.0]

    gs = gridspec.GridSpec(
        4, N, figure=fig,
        height_ratios=row_heights,
        hspace=0.55, wspace=0.35,
        top=0.97, bottom=0.04,
    )

    for col, enc in enumerate(enc_data):
        h, y, w, stats = enc["h"], enc["y"], enc["w"], enc["stats"]

        ax0 = fig.add_subplot(gs[0, col])
        _row_pca(ax0, h, y, stats["w_unit"], stats["r_corr"],
                 enc["label"], reducer=reducer)
        if col == 0:
            ax0.legend(markerscale=4, fontsize=7, loc="upper right")

        ax1 = fig.add_subplot(gs[1, col])
        _row_hist1d(ax1, stats["proj_1d"], y)
        if col == 0:
            ax1.legend(fontsize=7)

        ax2 = fig.add_subplot(gs[2, col])
        _row_delta_weights(ax2, stats["delta"], w, stats["r_corr"])

        ax3 = fig.add_subplot(gs[3, col])
        _row_active_dims(ax3, stats["active"], y, h.shape[1])
        if col == 0:
            ax3.legend(fontsize=7)

    # Row labels on the left side of the first column
    for row_i, rl in enumerate(row_labels):
        axes_row = [fig.axes[col * 4 + row_i] if False else None for col in range(N)]
    # Add row annotations as figure-level text
    y_positions = [0.955, 0.710, 0.500, 0.275]
    for yi, rl in zip(y_positions, row_labels):
        fig.text(0.005, yi, rl, va="center", ha="left",
                 fontsize=8, color="#555", rotation=90,
                 transform=fig.transFigure)

    return fig


def make_summary_figure(enc_data: list[dict]) -> plt.Figure:
    """Bar chart comparing r-correlation values across encodings."""
    labels = [e["label"] for e in enc_data]
    r_vals = [e["stats"]["r_corr"] for e in enc_data]

    fig, ax = plt.subplots(figsize=(max(5, 2 * len(labels)), 4))
    colors = [BLUE if r > 0 else RED for r in r_vals]
    bars = ax.bar(labels, r_vals, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Pearson r (activation delta vs probe weight)", fontsize=10)
    ax.set_title("Probe-delta correlation by encoding  (higher = more linearly decodable)",
                 fontsize=10)
    for bar, val in zip(bars, r_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01 * np.sign(val),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")
    ax.set_ylim(min(r_vals) - 0.1, max(r_vals) + 0.15)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to config JSON file")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory (default: directory of --config)")
    p.add_argument("--reducer", default="pca", choices=["pca", "umap"],
                   help="Dimensionality reduction for scatter (default: pca; umap requires umap-learn)")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    outdir = Path(args.output_dir) if args.output_dir else cfg_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path) as f:
        cfg = json.load(f)

    encodings   = cfg["encodings"]
    n_samp_ms   = int(cfg.get("n_samples_ms", 5000))
    n_samp_pb   = int(cfg.get("n_samples_pb", 0))
    reducer     = args.reducer

    print("=" * 64)
    print(f"  Mechanistic comparison: {len(encodings)} encodings")
    print("=" * 64)

    for dataset_key, n_samp, title_suffix, out_stem in [
        ("ms", n_samp_ms, "Math-Shepherd eval",  "comparison_ms"),
        ("pb", n_samp_pb, "ProcessBench GSM8K",  "comparison_pb"),
    ]:
        ms_key = "ms" if dataset_key == "ms" else "pb"
        print(f"\n--- {title_suffix} ---")

        enc_data = []
        for enc in encodings:
            npz_path  = enc[f"{ms_key}_npz"]
            label_key = enc.get(f"{ms_key}_label_key",
                                "correctness" if ms_key == "ms" else "step_labels")

            result = load_encoding(npz_path, label_key, n_samp)
            if result is None:
                print(f"  Skipping '{enc['label']}' for {title_suffix}")
                continue
            h, y = result

            w, b = load_probe_weights(enc["probes"])
            if w is None:
                print(f"  No valid probes for '{enc['label']}' -- skipping")
                continue

            stats = compute_stats(h, y, w)
            enc_data.append({"label": enc["label"], "h": h, "y": y, "w": w, "stats": stats})
            print(f"  {enc['label']:20s}  r={stats['r_corr']:.4f}  "
                  f"active_dims_mean={stats['active'].mean():.1f}")

        if not enc_data:
            print(f"  No encodings available for {title_suffix} -- skipping figure")
            continue

        print(f"\n  Generating figure ({len(enc_data)} columns, reducer={reducer})...")
        fig = make_comparison_figure(
            enc_data,
            dataset_title=f"Encoding comparison — {title_suffix}  "
                          f"(n={len(enc_data[0]['h']):,} balanced steps per encoding)",
            reducer=reducer,
        )
        out = outdir / f"{out_stem}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

        # Summary r-bar chart for this dataset
        fig_r = make_summary_figure(enc_data)
        out_r = outdir / f"{out_stem}_r_summary.png"
        fig_r.savefig(out_r, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_r)
        print(f"  Saved: {out_r}")

    print("\n=== Done ===")
    print(f"  Figures in: {outdir}")
    print("  Fetch locally with:")
    print(f"  rsync -avz $USER@tamia.alliancecan.ca:~/CoT-checker/results/mechanistic_comparison/ "
          f"./results/mechanistic_comparison/")


if __name__ == "__main__":
    main()
