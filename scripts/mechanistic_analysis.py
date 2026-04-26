#!/usr/bin/env python3
"""Mechanistic analysis of SSAE latents from the experiment-7 encodings.

Runs entirely on CPU from pre-encoded .npz files and saved linear probe
checkpoints.  No SSAE model or GPU required.

Outputs:
  - Textual summary to stdout (suitable for cluster log capture)
  - Plots saved to --output-dir (default: results/mechanistic/)

Usage:
    python scripts/mechanistic_analysis.py \
        --eval-data  $STORE/probe_data/eval_held_out.npz \
        --train-data $STORE/probe_data/train_final.npz \
        --probes     $STORE/results/linear_probe_seed42.pt \
                     $STORE/results/linear_probe_seed43.pt \
                     $STORE/results/linear_probe_seed44.pt \
                     $STORE/results/linear_probe_seed45.pt \
        --output-dir results/mechanistic/
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(path: str, max_samples: int | None = None, seed: int = 42):
    d = np.load(path)
    h = d["latents"].astype(np.float32)
    y = d["correctness"].astype(np.int32)
    if max_samples and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        h, y = h[idx], y[idx]
    print(f"  Loaded {len(y):,} steps from {path}  (dim={h.shape[1]})")
    print(f"    correct: {(y==1).sum():,}  incorrect: {(y==0).sum():,}")
    return h, y


def load_linear_probe(path: str) -> np.ndarray:
    """Return the weight vector w (shape: [D]) and bias b (scalar)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["state_dict"]
    w = sd["fc.weight"].numpy().squeeze()  # (D,)
    b = sd["fc.bias"].numpy().squeeze()    # scalar
    return w, b


def mean_cosine_sim(A: np.ndarray, B: np.ndarray, n_pairs: int = 10_000, seed: int = 42) -> float:
    """Estimate mean cosine similarity between random pairs from A and B.
    Assumes vectors are already L2-normalized."""
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(A), size=min(n_pairs, len(A)), replace=False)
    ib = rng.choice(len(B), size=min(n_pairs, len(B)), replace=False)
    return float((A[ia] * B[ib]).sum(axis=1).mean())


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_sparsity(h: np.ndarray, y: np.ndarray) -> dict:
    """Active-dimension statistics per class."""
    section("1. Sparsity statistics (active dimensions per step)")
    D = h.shape[1]
    active = (h > 0).sum(axis=1)  # (N,)
    cor = active[y == 1]
    inc = active[y == 0]
    out = {}
    for name, arr in [("correct", cor), ("incorrect", inc)]:
        out[name] = {
            "mean": arr.mean(), "std": arr.std(),
            "median": np.median(arr), "min": arr.min(), "max": arr.max(),
            "pct25": np.percentile(arr, 25), "pct75": np.percentile(arr, 75),
        }
        print(f"\n  {name.capitalize()} steps (n={len(arr):,}):")
        print(f"    mean active dims : {arr.mean():.1f} / {D}  ({100*arr.mean()/D:.1f}%)")
        print(f"    std              : {arr.std():.1f}")
        print(f"    median           : {np.median(arr):.0f}")
        print(f"    25th–75th pct    : {np.percentile(arr, 25):.0f} – {np.percentile(arr, 75):.0f}")
    delta = out["incorrect"]["mean"] - out["correct"]["mean"]
    print(f"\n  Delta (incorrect - correct) : {delta:.1f} active dims")
    out["delta"] = delta
    return out


def analyze_probe_weights(weights: list[tuple[np.ndarray, float]], D: int) -> np.ndarray:
    """Average probe weights across seeds and show top dimensions."""
    section("2. Linear probe weight analysis")
    ws = np.stack([w for w, _ in weights], axis=0)  # (S, D)
    bs = np.array([b for _, b in weights])
    w_mean = ws.mean(axis=0)  # (D,)
    w_std  = ws.std(axis=0)
    print(f"\n  Seeds: {len(weights)}   |w| stats across dimensions:")
    print(f"    mean |w|  : {np.abs(w_mean).mean():.6f}")
    print(f"    max  |w|  : {np.abs(w_mean).max():.6f}")
    print(f"    95th pct  : {np.percentile(np.abs(w_mean), 95):.6f}")
    print(f"    mean bias : {bs.mean():.4f} ± {bs.std():.4f}")
    print(f"\n  Fraction of dims with |w| > 0.01 : "
          f"{(np.abs(w_mean) > 0.01).sum():,} / {D}  "
          f"({100*(np.abs(w_mean) > 0.01).mean():.1f}%)")

    top_pos = np.argsort(w_mean)[::-1][:20]
    top_neg = np.argsort(w_mean)[:20]

    print("\n  Top-20 dims that predict CORRECT (positive weight):")
    print(f"  {'dim':>6}  {'w_mean':>10}  {'w_std':>8}")
    for d in top_pos:
        print(f"  {d:>6}  {w_mean[d]:>10.6f}  {w_std[d]:>8.6f}")

    print("\n  Top-20 dims that predict INCORRECT (negative weight):")
    print(f"  {'dim':>6}  {'w_mean':>10}  {'w_std':>8}")
    for d in top_neg:
        print(f"  {d:>6}  {w_mean[d]:>10.6f}  {w_std[d]:>8.6f}")

    return w_mean


def analyze_per_feature(h: np.ndarray, y: np.ndarray, w_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-dimension mean activations and correlation with probe weights."""
    section("3. Per-feature activation analysis")
    D = h.shape[1]
    cor_mask = y == 1
    inc_mask = y == 0

    mean_cor = h[cor_mask].mean(axis=0)   # (D,)
    mean_inc = h[inc_mask].mean(axis=0)   # (D,)
    delta = mean_cor - mean_inc            # positive = more active for correct

    # Active rates
    rate_cor = (h[cor_mask] > 0).mean(axis=0)   # (D,)
    rate_inc = (h[inc_mask] > 0).mean(axis=0)   # (D,)

    # Correlation between probe weights and mean-activation delta
    corr = np.corrcoef(w_mean, delta)[0, 1]
    print(f"\n  Pearson correlation between probe weights and (mean_correct - mean_incorrect):")
    print(f"    r = {corr:.4f}")
    print(f"  (r=1.0 means the probe exactly tracks the mean-activation difference)")

    # Top features by |delta|
    top_delta = np.argsort(np.abs(delta))[::-1][:30]
    print(f"\n  Top-30 features by |mean_correct - mean_incorrect|:")
    print(f"  {'dim':>6}  {'mean_cor':>10}  {'mean_inc':>10}  {'delta':>10}  "
          f"{'rate_cor':>10}  {'rate_inc':>10}  {'probe_w':>10}")
    for d in top_delta:
        print(f"  {d:>6}  {mean_cor[d]:>10.6f}  {mean_inc[d]:>10.6f}  "
              f"{delta[d]:>10.6f}  {rate_cor[d]:>10.4f}  {rate_inc[d]:>10.4f}  "
              f"{w_mean[d]:>10.6f}")

    # Dims where activation is consistently zero for correct but non-zero for incorrect
    zero_cor_nonzero_inc = (rate_cor < 0.01) & (rate_inc > 0.05)
    zero_inc_nonzero_cor = (rate_inc < 0.01) & (rate_cor > 0.05)
    print(f"\n  Dims near-zero for correct but active for incorrect (rate_cor<1%, rate_inc>5%): "
          f"{zero_cor_nonzero_inc.sum()}")
    print(f"  Dims near-zero for incorrect but active for correct (rate_inc<1%, rate_cor>5%): "
          f"{zero_inc_nonzero_cor.sum()}")

    return mean_cor, mean_inc, delta


def analyze_geometry(h: np.ndarray, y: np.ndarray) -> dict:
    """Cosine similarity structure on the unit sphere."""
    section("4. Geometric analysis (unit sphere cosine similarities)")
    # h should already be L2-normalized from the SSAE encoder
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    h_norm = h / np.clip(norms, 1e-9, None)

    cor = h_norm[y == 1]
    inc = h_norm[y == 0]

    sim_within_cor = mean_cosine_sim(cor, cor, n_pairs=20_000)
    sim_within_inc = mean_cosine_sim(inc, inc, n_pairs=20_000)
    sim_between    = mean_cosine_sim(cor, inc, n_pairs=20_000)

    print(f"\n  Mean cosine similarity within correct class  : {sim_within_cor:.4f}")
    print(f"  Mean cosine similarity within incorrect class: {sim_within_inc:.4f}")
    print(f"  Mean cosine similarity between classes       : {sim_between:.4f}")
    print(f"\n  Separation gap = within_correct - between: {sim_within_cor - sim_between:+.4f}")
    print(f"  Separation gap = within_incorrect - between: {sim_within_inc - sim_between:+.4f}")

    return {
        "within_correct": sim_within_cor,
        "within_incorrect": sim_within_inc,
        "between": sim_between,
    }


def _numpy_pca(X: np.ndarray, n_components: int = 10):
    """Truncated PCA via SVD. Returns (Z, components, explained_variance_ratio)."""
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    var = (S ** 2) / (len(X) - 1)
    total_var = var.sum()
    components = Vt[:n_components]          # (n_components, D)
    Z = X_centered @ components.T          # (N, n_components)
    explained = var[:n_components] / total_var
    return Z, components, explained


def pca_analysis(h: np.ndarray, y: np.ndarray, w_mean: np.ndarray, outdir: Path) -> None:
    """PCA visualization and probe direction projection."""
    section("5. PCA analysis")
    n_components = 10

    # Subsample for speed if large
    max_pca = 20_000
    if len(y) > max_pca:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y), size=max_pca, replace=False)
        h_sub, y_sub = h[idx], y[idx]
    else:
        h_sub, y_sub = h, y

    Z, components, explained = _numpy_pca(h_sub, n_components)

    print(f"\n  PCA ({n_components} components):")
    print(f"  {'PC':>4}  {'var_explained':>15}  {'cumulative':>12}")
    cum = 0.0
    for i, v in enumerate(explained):
        cum += v
        print(f"  {i+1:>4}  {v*100:>14.2f}%  {cum*100:>11.2f}%")

    # Project probe weight direction into PCA space
    w_unit = w_mean / (np.linalg.norm(w_mean) + 1e-9)
    w_proj = components @ w_unit  # (n_components,)
    print(f"\n  Probe weight direction projected onto PC1–PC5:")
    for i in range(5):
        print(f"    PC{i+1}: {w_proj[i]:+.6f}")

    # --- Plot 1: PC1 vs PC2 scatter, colored by label ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color, name in [(1, "#2196F3", "correct"), (0, "#F44336", "incorrect")]:
        mask = y_sub == label
        ax.scatter(Z[mask, 0], Z[mask, 1], c=color, alpha=0.15, s=2, label=name, rasterized=True)
    # overlay probe direction (projected onto PC1-PC2 plane)
    scale = np.percentile(np.abs(Z[:, :2]), 90)
    dx, dy = w_proj[0] * scale, w_proj[1] * scale
    ax.annotate(
        "", xy=(dx, dy), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )
    ax.text(dx * 1.05, dy * 1.05, "probe direction", fontsize=9, ha="center")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title("SSAE latents: PC1 vs PC2 (correct / incorrect)")
    ax.legend(markerscale=5, loc="upper right")
    plt.tight_layout()
    out = outdir / "pca_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out}")

    # --- Plot 2: 1D projection onto probe direction ---
    w_unit = w_mean / (np.linalg.norm(w_mean) + 1e-9)
    proj_1d = h_sub @ w_unit  # (N,)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(proj_1d.min(), proj_1d.max(), 80)
    ax.hist(proj_1d[y_sub == 1], bins=bins, alpha=0.6, color="#2196F3",
            label="correct", density=True)
    ax.hist(proj_1d[y_sub == 0], bins=bins, alpha=0.6, color="#F44336",
            label="incorrect", density=True)
    ax.set_xlabel("Projection onto probe direction (w^T h_c)")
    ax.set_ylabel("Density")
    ax.set_title("1D projection: correct vs incorrect along probe axis")
    ax.legend()
    plt.tight_layout()
    out = outdir / "probe_projection_1d.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sparsity_hist(h: np.ndarray, y: np.ndarray, outdir: Path) -> None:
    active = (h > 0).sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(active.min(), active.max() + 2)
    ax.hist(active[y == 1], bins=bins, alpha=0.6, color="#2196F3",
            label="correct", density=True)
    ax.hist(active[y == 0], bins=bins, alpha=0.6, color="#F44336",
            label="incorrect", density=True)
    ax.set_xlabel("Number of active dimensions (out of 896)")
    ax.set_ylabel("Density")
    ax.set_title("Active dimension count per step: correct vs incorrect")
    ax.legend()
    plt.tight_layout()
    out = outdir / "active_dims_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out}")


def plot_delta_vs_weights(delta: np.ndarray, w_mean: np.ndarray, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(delta, w_mean, alpha=0.3, s=3, color="#555", rasterized=True)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("mean_correct[i] - mean_incorrect[i]  (activation delta)")
    ax.set_ylabel("Probe weight w[i]")
    ax.set_title("Per-feature: activation delta vs probe weight")
    corr = np.corrcoef(delta, w_mean)[0, 1]
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            va="top", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = outdir / "delta_vs_weights.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_top_features(mean_cor: np.ndarray, mean_inc: np.ndarray,
                      w_mean: np.ndarray, outdir: Path, k: int = 40) -> None:
    """Bar chart: top-k features sorted by |activation delta|."""
    delta = mean_cor - mean_inc
    top_idx = np.argsort(np.abs(delta))[::-1][:k]
    x = np.arange(k)
    width = 0.3

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Upper: mean activations
    ax = axes[0]
    ax.bar(x - width/2, mean_cor[top_idx], width, color="#2196F3", label="correct")
    ax.bar(x + width/2, mean_inc[top_idx], width, color="#F44336", label="incorrect")
    ax.set_ylabel("Mean activation")
    ax.set_title(f"Top-{k} features by |activation delta|: mean activations")
    ax.legend()

    # Lower: probe weights
    ax = axes[1]
    colors = ["#4CAF50" if v > 0 else "#FF5722" for v in w_mean[top_idx]]
    ax.bar(x, w_mean[top_idx], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Feature rank (by |activation delta|)")
    ax.set_ylabel("Probe weight w[i]")
    ax.set_title("Probe weights for the same features")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(i) for i in top_idx], rotation=90, fontsize=6)

    plt.tight_layout()
    out = outdir / "top_features.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Mechanistic analysis of SSAE latents")
    p.add_argument("--eval-data", required=True,
                   help="Path to eval_held_out.npz (50K balanced steps)")
    p.add_argument("--train-data", default=None,
                   help="Optional: path to train_final.npz for sparsity stats on larger pool")
    p.add_argument("--probes", nargs="+", required=True,
                   help="Paths to linear_probe_seed*.pt checkpoints")
    p.add_argument("--output-dir", default="results/mechanistic/")
    p.add_argument("--max-train-samples", type=int, default=50_000,
                   help="Max samples to load from train-data for stats (default: 50K for speed)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Mechanistic Analysis: SSAE Latents (experiment-7 encodings)")
    print("=" * 70)

    # --- Load eval data ---
    print(f"\n[1/7] Loading eval data...")
    h_eval, y_eval = load_npz(args.eval_data)

    # --- Load probe weights ---
    print(f"\n[2/7] Loading {len(args.probes)} linear probe(s)...")
    probe_weights = []
    for path in args.probes:
        w, b = load_linear_probe(path)
        probe_weights.append((w, b))
        print(f"  Loaded {path}  (D={len(w)})")

    D = h_eval.shape[1]

    # --- Optionally load training data for additional sparsity stats ---
    h_train, y_train = None, None
    if args.train_data:
        print(f"\n[3/7] Loading training data (max {args.max_train_samples:,} samples)...")
        h_train, y_train = load_npz(args.train_data, max_samples=args.max_train_samples)

    # -----------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  ANALYSIS ON EVAL SET  (N={len(y_eval):,}, balanced 50/50)")
    print(f"{'─'*70}")

    sparsity_stats = analyze_sparsity(h_eval, y_eval)
    plot_sparsity_hist(h_eval, y_eval, outdir)

    if h_train is not None:
        print(f"\n{'─'*70}")
        print(f"  SPARSITY STATS ON TRAINING POOL  (N={len(y_train):,})")
        print(f"{'─'*70}")
        analyze_sparsity(h_train, y_train)

    w_mean = analyze_probe_weights(probe_weights, D)
    mean_cor, mean_inc, delta = analyze_per_feature(h_eval, y_eval, w_mean)

    plot_delta_vs_weights(delta, w_mean, outdir)
    plot_top_features(mean_cor, mean_inc, w_mean, outdir)

    analyze_geometry(h_eval, y_eval)

    pca_analysis(h_eval, y_eval, w_mean, outdir)

    # -----------------------------------------------------------------------
    section("Summary")
    print(f"\n  Eval set: {len(y_eval):,} steps  ({(y_eval==1).sum():,} correct, {(y_eval==0).sum():,} incorrect)")
    print(f"  Latent dimensionality: {D}")
    print(f"\n  Sparsity (eval):")
    print(f"    Correct   : {sparsity_stats['correct']['mean']:.1f} ± {sparsity_stats['correct']['std']:.1f} active dims")
    print(f"    Incorrect : {sparsity_stats['incorrect']['mean']:.1f} ± {sparsity_stats['incorrect']['std']:.1f} active dims")
    print(f"    Delta     : {sparsity_stats['delta']:+.1f} dims  (incorrect - correct)")
    print(f"\n  Probe weight range: [{w_mean.min():.5f}, {w_mean.max():.5f}]")
    print(f"  Probe-delta correlation: r = {np.corrcoef(w_mean, delta)[0,1]:.4f}")
    print(f"\n  Output plots: {outdir}/")
    print(f"    active_dims_hist.png  — sparsity distribution per class")
    print(f"    delta_vs_weights.png  — activation delta vs probe weight (scatter)")
    print(f"    top_features.png      — top-40 discriminating features")
    print(f"    pca_scatter.png       — PC1 vs PC2 colored by label + probe direction")
    print(f"    probe_projection_1d.png — 1D projection histogram along probe axis")
    print("\nDone.")


if __name__ == "__main__":
    main()
