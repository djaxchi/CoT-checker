#!/usr/bin/env python3
"""Transition probe analysis: train probes on Δz = z_k - z_{k-1} representations.

For each encoding, computes delta representations between consecutive steps within
the same solution, trains linear probes on those deltas, and compares mechanistic
alignment (r) against the per-step baseline.

Requires npz files with solution_ids + step_positions arrays. ProcessBench npz
files have these by default. Math-Shepherd npz files need generate_probe_data.py
run with --save-meta (or a recent version that always saves these fields).

Config JSON format: same as compare_mechanistic_viz.py.

Output figures:
  transition_comparison_ms.png    -- 4-row × N-col mechanistic figure (Math-Shepherd)
  transition_comparison_pb.png    -- same for ProcessBench
  transition_r_summary_ms.png     -- grouped bar: per-step r vs transition r (MS)
  transition_r_summary_pb.png     -- same for ProcessBench

Usage:
  python scripts/analyze_transition_probes.py \\
      --config results/mechanistic_comparison/config.json \\
      --output-dir results/mechanistic_comparison/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BLUE = "#1565C0"
RED  = "#C62828"
GREY = "#424242"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def compute_deltas(npz_path: str, label_key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load npz and compute z_k - z_{k-1} within each solution.

    Returns (delta_latents float32, delta_labels int32) sorted by (solution_id,
    step_position), or None if the file is missing or lacks sequence metadata.
    """
    p = Path(npz_path)
    if not p.exists():
        print(f"  WARNING: {p.name} not found -- skipping")
        return None
    d = np.load(p)
    if "solution_ids" not in d or "step_positions" not in d:
        print(f"  WARNING: {p.name} lacks solution_ids/step_positions -- skipping transition")
        return None

    latents      = d["latents"].astype(np.float32)
    labels       = d[label_key].astype(np.int32)
    solution_ids = d["solution_ids"]
    step_pos     = d["step_positions"].astype(np.int32)

    order = np.lexsort((step_pos, solution_ids))
    latents      = latents[order]
    labels       = labels[order]
    solution_ids = solution_ids[order]
    step_pos     = step_pos[order]

    deltas, dlabels = [], []
    for i in range(1, len(latents)):
        if solution_ids[i] != solution_ids[i - 1]:
            continue
        deltas.append(latents[i] - latents[i - 1])
        dlabels.append(labels[i])

    if not deltas:
        print(f"  WARNING: no valid delta pairs in {p.name}")
        return None

    delta_arr = np.array(deltas, dtype=np.float32)
    label_arr = np.array(dlabels, dtype=np.int32)
    n1, n0 = (label_arr == 1).sum(), (label_arr == 0).sum()
    print(f"  {p.name}: {len(delta_arr):,} delta pairs  "
          f"correct={n1}  incorrect={n0}  dim={delta_arr.shape[1]}")
    return delta_arr, label_arr


def load_per_step(npz_path: str, label_key: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = Path(npz_path)
    if not p.exists():
        return None
    d = np.load(p)
    return d["latents"].astype(np.float32), d[label_key].astype(np.int32)


def balance(X: np.ndarray, y: np.ndarray, n: int = 0, seed: int = 42
            ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    k = min(len(idx1), len(idx0))
    if n > 0:
        k = min(k, n // 2)
    sel = np.concatenate([rng.choice(idx1, k, replace=False),
                          rng.choice(idx0, k, replace=False)])
    rng.shuffle(sel)
    return X[sel], y[sel]


def train_linear_probe(X: np.ndarray, y: np.ndarray,
                       seeds: tuple = (42, 43, 44, 45),
                       val_frac: float = 0.2) -> tuple[np.ndarray, float]:
    """Return (mean_weights, mean_val_accuracy) across seeds."""
    rng = np.random.default_rng(0)
    n_val = max(2, int(len(X) * val_frac))
    idx = rng.permutation(len(X))
    X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]
    X_tr,  y_tr  = X[idx[n_val:]], y[idx[n_val:]]
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    ws, accs = [], []
    for seed in seeds:
        clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0, solver="lbfgs")
        clf.fit(X_tr_s, y_tr)
        ws.append(clf.coef_[0])
        accs.append(clf.score(X_val_s, y_val))
    return np.stack(ws).mean(0), float(np.mean(accs))


def load_probe_weights(probe_paths: list[str]) -> np.ndarray | None:
    ws = []
    for path in probe_paths:
        p = Path(path)
        if not p.exists():
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        ws.append(ckpt["state_dict"]["fc.weight"].numpy().squeeze())
    return np.stack(ws).mean(0) if ws else None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    w_unit  = w / (np.linalg.norm(w) + 1e-9)
    mean_c  = X[y == 1].mean(0)
    mean_i  = X[y == 0].mean(0)
    delta   = mean_c - mean_i
    r_corr  = float(np.corrcoef(w, delta)[0, 1])
    proj_1d = X @ w_unit
    # "changed dims": count dims where |value| > 0. For delta reps this means
    # the feature activation shifted; for per-step reps it's the usual positive count.
    active  = (np.abs(X) > 1e-4).sum(1)
    return {"delta": delta, "r_corr": r_corr, "proj_1d": proj_1d,
            "active": active, "w_unit": w_unit}


def truncated_pca(X: np.ndarray, n: int = 2):
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = (S ** 2).sum()
    expl = (S[:n] ** 2) / total_var
    Z = Xc @ Vt[:n].T
    return Z, Vt[:n], expl


# ---------------------------------------------------------------------------
# Figure rows (mirrors compare_mechanistic_viz.py style)
# ---------------------------------------------------------------------------

def _row_pca(ax, X, y, w_unit, r_corr, label, n_scatter=2000):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(n_scatter, len(X)), replace=False)
    Xs = X[idx]; ys = y[idx]
    Z, comps, expl = truncated_pca(Xs)
    for cls, color, name in [(1, BLUE, "correct"), (0, RED, "incorrect")]:
        m = ys == cls
        ax.scatter(Z[m, 0], Z[m, 1], c=color, alpha=0.25, s=4,
                   label=name, rasterized=True)
    w_proj = comps @ w_unit
    if (np.abs(w_proj) > 1e-6).any():
        scale = np.percentile(np.abs(Z), 85)
        ax.annotate("", xy=(w_proj[0]*scale, w_proj[1]*scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2.0))
    ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}%)", fontsize=8)
    ax.set_title(f"{label}\nr={r_corr:.3f}", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)


def _row_hist1d(ax, proj_1d, y, xlabel="w·Δz"):
    bins = np.linspace(proj_1d.min(), proj_1d.max(), 55)
    ax.hist(proj_1d[y == 1], bins=bins, density=True, alpha=0.65, color=BLUE, label="correct")
    ax.hist(proj_1d[y == 0], bins=bins, density=True, alpha=0.65, color=RED,  label="incorrect")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)


def _row_delta_weights(ax, delta, w, r_corr, xlabel="Δactivation (cor−inc)"):
    ax.scatter(delta, w, alpha=0.25, s=2, color=GREY, rasterized=True)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("probe weight w[i]", fontsize=8)
    ax.text(0.05, 0.95, f"r = {r_corr:.3f}", transform=ax.transAxes,
            va="top", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)


def _row_active_dims(ax, active, y, D, xlabel="# changed dims"):
    lo, hi = int(active.min()), int(active.max())
    if lo == hi:
        ax.axvline(lo, color=BLUE, lw=2, label=f"all = {lo}")
        ax.set_xlim(max(0, lo - 5), min(D, lo + 5))
    else:
        bins = np.linspace(lo, hi, min(50, hi - lo + 2))
        ax.hist(active[y == 1], bins=bins, density=True, alpha=0.65, color=BLUE, label="correct")
        ax.hist(active[y == 0], bins=bins, density=True, alpha=0.65, color=RED,  label="incorrect")
    ax.set_xlabel(f"{xlabel}  (D={D})", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Full comparison figure
# ---------------------------------------------------------------------------

def make_transition_figure(enc_data: list[dict], title: str,
                           is_delta: bool = True) -> plt.Figure:
    N = len(enc_data)
    fig = plt.figure(figsize=(5 * N, 17))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.998)
    gs = gridspec.GridSpec(4, N, figure=fig,
                           height_ratios=[4.5, 3.0, 3.5, 3.0],
                           hspace=0.55, wspace=0.35, top=0.97, bottom=0.04)

    proj_label  = "w·Δz (probe score)" if is_delta else "w·z (probe score)"
    delta_xlabel = "Δ(cor−inc) transitions" if is_delta else "Δactivation (cor−inc)"
    dims_xlabel  = "# changed dims" if is_delta else "# active dims"

    for col, enc in enumerate(enc_data):
        X, y, w, stats = enc["X"], enc["y"], enc["w"], enc["stats"]

        ax0 = fig.add_subplot(gs[0, col])
        _row_pca(ax0, X, y, stats["w_unit"], stats["r_corr"], enc["label"])
        if col == 0:
            ax0.legend(markerscale=4, fontsize=7, loc="upper right")

        ax1 = fig.add_subplot(gs[1, col])
        _row_hist1d(ax1, stats["proj_1d"], y, xlabel=proj_label)
        if col == 0:
            ax1.legend(fontsize=7)

        ax2 = fig.add_subplot(gs[2, col])
        _row_delta_weights(ax2, stats["delta"], w, stats["r_corr"], xlabel=delta_xlabel)

        ax3 = fig.add_subplot(gs[3, col])
        _row_active_dims(ax3, stats["active"], y, X.shape[1], xlabel=dims_xlabel)
        if col == 0:
            ax3.legend(fontsize=7)

    row_labels = ["PCA scatter", "1D probe projection",
                  "Δactivation vs probe weight", "Active / changed dims"]
    for yi, rl in zip([0.955, 0.710, 0.500, 0.275], row_labels):
        fig.text(0.005, yi, rl, va="center", ha="left", fontsize=8,
                 color="#555", rotation=90, transform=fig.transFigure)
    return fig


# ---------------------------------------------------------------------------
# R-comparison summary figure
# ---------------------------------------------------------------------------

def make_r_comparison_figure(labels: list[str],
                             r_step: list[float], r_trans: list[float],
                             title: str) -> plt.Figure:
    x = np.arange(len(labels))
    bw = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(labels)), 5))
    b1 = ax.bar(x - bw/2, r_step,  bw, label="per-step probe",       color="#1565C0", alpha=0.85)
    b2 = ax.bar(x + bw/2, r_trans, bw, label="transition probe (Δz)", color="#E65100", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Pearson r (activation delta vs probe weight)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ymax = max(max(r_step + r_trans), 0.05) * 1.3 + 0.05
    ax.set_ylim(0, ymax)
    for bars, vals in [(b1, r_step), (b2, r_trans)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(encodings: list[dict], dataset_key: str, n_samples: int,
                    outdir: Path, dpi: int, title_suffix: str) -> None:
    npz_key   = f"{dataset_key}_npz"
    lkey_cfg  = f"{dataset_key}_label_key"
    default_lkey = "correctness" if dataset_key == "ms" else "step_labels"

    enc_trans, r_step_list, r_trans_list, labels_list = [], [], [], []

    for enc in encodings:
        label    = enc["label"]
        npz_path = enc[npz_key]
        lkey     = enc.get(lkey_cfg, default_lkey)

        # --- Transition deltas ---
        result = compute_deltas(npz_path, lkey)
        if result is None:
            continue
        delta_X, delta_y = result
        if len(delta_X) < 30:
            print(f"  {label}: only {len(delta_X)} delta pairs -- too few, skipping")
            continue

        X_b, y_b = balance(delta_X, delta_y, n=n_samples)
        print(f"  {label}: training on {len(X_b)} balanced delta pairs")

        w_trans, acc_trans = train_linear_probe(X_b, y_b)
        stats_trans = compute_stats(X_b, y_b, w_trans)
        enc_trans.append({"label": label, "X": X_b, "y": y_b,
                          "w": w_trans, "stats": stats_trans})

        # --- Per-step r (from existing probe checkpoints) ---
        w_step = load_probe_weights(enc.get("probes", []))
        per_step = load_per_step(npz_path, lkey)
        if w_step is not None and per_step is not None:
            h, yh = balance(per_step[0], per_step[1], n=n_samples)
            if w_step.shape[0] == h.shape[1]:
                stats_step = compute_stats(h, yh, w_step)
                r_step_list.append(stats_step["r_corr"])
                r_trans_list.append(stats_trans["r_corr"])
                labels_list.append(label)
                print(f"  {label:22s}  per-step r={stats_step['r_corr']:.3f}  "
                      f"trans r={stats_trans['r_corr']:.3f}  "
                      f"trans acc={acc_trans:.3f}")
            else:
                print(f"  {label}: probe dim mismatch ({w_step.shape[0]} vs {h.shape[1]}) -- "
                      f"trans r={stats_trans['r_corr']:.3f}")
        else:
            print(f"  {label}: no per-step probes  trans r={stats_trans['r_corr']:.3f}  "
                  f"acc={acc_trans:.3f}")

    if not enc_trans:
        print(f"  No encodings processed for {title_suffix}")
        return

    n_pairs = len(enc_trans[0]["X"])
    fig = make_transition_figure(
        enc_trans,
        title=(f"Transition probe comparison — {title_suffix}  "
               f"(n={n_pairs:,} balanced Δz pairs per encoding)"),
        is_delta=True,
    )
    out = outdir / f"transition_comparison_{dataset_key}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    if labels_list:
        fig_r = make_r_comparison_figure(
            labels_list, r_step_list, r_trans_list,
            title=f"Per-step vs transition probe — {title_suffix}",
        )
        out_r = outdir / f"transition_r_summary_{dataset_key}.png"
        fig_r.savefig(out_r, dpi=dpi, bbox_inches="tight")
        plt.close(fig_r)
        print(f"  Saved: {out_r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True, help="Path to config.json")
    p.add_argument("--output-dir",  default=None)
    p.add_argument("--n-samples",   type=int, default=0,
                   help="Max balanced pairs per encoding (0=all)")
    p.add_argument("--dpi",         type=int, default=150)
    return p.parse_args()


def main():
    args     = parse_args()
    cfg_path = Path(args.config)
    outdir   = Path(args.output_dir) if args.output_dir else cfg_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path) as f:
        cfg = json.load(f)

    encodings = cfg["encodings"]
    n_samp_ms = args.n_samples or int(cfg.get("n_samples_ms", 0))
    n_samp_pb = args.n_samples or int(cfg.get("n_samples_pb", 0))

    print("=" * 64)
    print(f"  Transition probe analysis: {len(encodings)} encodings")
    print("=" * 64)

    print("\n--- ProcessBench GSM8K ---")
    process_dataset(encodings, "pb", n_samp_pb, outdir, args.dpi, "ProcessBench GSM8K")

    print("\n--- Math-Shepherd eval ---")
    process_dataset(encodings, "ms", n_samp_ms, outdir, args.dpi, "Math-Shepherd eval")

    print("\n=== Done ===")
    print(f"  Figures in: {outdir}")
    print("  Fetch locally:")
    print("  rsync -avz $USER@tamia.alliancecan.ca:"
          "~/CoT-checker/results/mechanistic_comparison/ ./results/mechanistic_comparison/")


if __name__ == "__main__":
    main()
