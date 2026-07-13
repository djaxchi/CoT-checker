"""Layer trajectories of the last-token representation: do correct and incorrect steps
travel different paths through the network?

Hypothesis (Djalil): instead of reading a single (layer, token) plane, summarise the
*trajectory* of the candidate step's hidden state across layers with one number per layer
and ask whether that curve separates correct from incorrect steps.

Two per-layer summaries are computed (chosen with the user):
  - raw_max : max over hidden dims of the signed activation  (exposed to "massive
              activation" / rogue dims that grow with depth, content-independent)
  - l2_norm : L2 norm of the layer representation             (overall activation scale)

Each step becomes a length-L trajectory (L = stored layers). We plot, per metric:
  (1) mean trajectory +/- 95% CI, correct vs incorrect, raw scale
  (2) the same after per-layer z-scoring, so the *shape* difference is visible even
      though raw norm/max grow by orders of magnitude with depth
  (3) a faint sample of individual trajectories coloured by label
  (4) per-layer single-number discriminability: ROC-AUC and Cohen's d of that scalar

The point of (4) is the honest test: if the trajectory "looks" different only because of
the massive-activation dims, the per-layer AUC stays at chance and we learn that the raw
max is an artifact, not a correctness signal.

Inputs are the 4D multi-token/multi-layer encoding (n, L, T, H) already on disk; this is
pure numpy + matplotlib, no model, no GPU.

Outputs (results/prm800k_layers/trajectory/):
  - trajectory_{token}.png    4-panel summary (raw_max + l2_norm stacked)
  - trajectory_{token}.json   every number (per-layer AUC, Cohen's d, group means)

Usage:
    python scripts/analysis/s3_prm800k_layer_trajectory.py                 # last token
    python scripts/analysis/s3_prm800k_layer_trajectory.py --token first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.prm800k_val_data import load_prm800k_multitoken

DEFAULT_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b/prm_multitoken")
ROOT = Path("results/prm800k_layers/trajectory")
SEED = 42

GREEN = "#3cb44b"   # correct
RED = "#e6194B"     # incorrect


def metric_per_layer(H: np.ndarray, metric: str) -> np.ndarray:
    """(n, H) -> (n,) one summary number for this layer's representation."""
    if metric == "raw_max":
        return H.max(axis=1)
    if metric == "l2_norm":
        return np.linalg.norm(H, axis=1)
    raise ValueError(f"unknown metric {metric!r}")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardized mean difference (incorrect - correct)."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def directed_auc(score: np.ndarray, y: np.ndarray) -> float:
    """AUC for the single scalar predicting y=1 (incorrect), flipped to be >=0.5 so the
    number reports separability regardless of which way the feature points."""
    auc = roc_auc_score(y, score)
    return float(max(auc, 1.0 - auc))


def build_trajectories(d_by_layer, layers, metric):
    """-> traj (n, L) of the per-layer summary; rows aligned across layers."""
    cols = [metric_per_layer(d_by_layer[L].hidden, metric) for L in layers]
    return np.column_stack(cols).astype(np.float64)


def mean_ci(traj: np.ndarray):
    """Per-layer mean and 95% CI half-width (1.96 * SE) down the rows."""
    mu = traj.mean(axis=0)
    se = traj.std(axis=0, ddof=1) / np.sqrt(traj.shape[0])
    return mu, 1.96 * se


def plot_metric(axrow, layers, traj, y, metric, rng, n_sample):
    """Fill one row of 2 axes (raw mean+CI / z-scored mean+CI) and return per-layer stats.
    The individual-trajectory sample is drawn faintly on the z-scored panel."""
    x = np.asarray(layers)
    tc, ti = traj[y == 0], traj[y == 1]              # correct / incorrect

    # (left) raw mean trajectory +/- 95% CI
    for t, c, lab in [(tc, GREEN, "correct"), (ti, RED, "incorrect")]:
        mu, ci = mean_ci(t)
        axrow[0].plot(x, mu, "-o", color=c, label=lab)
        axrow[0].fill_between(x, mu - ci, mu + ci, color=c, alpha=0.2)
    axrow[0].set_title(f"{metric}: mean trajectory +/- 95% CI (raw)")
    axrow[0].set_xlabel("layer"); axrow[0].set_ylabel(metric); axrow[0].legend()

    # per-layer z-score (shared across both groups) so shape is comparable across depth
    mu_all = traj.mean(axis=0); sd_all = traj.std(axis=0) + 1e-9
    z = (traj - mu_all) / sd_all
    zc, zi = z[y == 0], z[y == 1]

    # faint individual sample on the z panel
    idx = rng.choice(len(z), size=min(n_sample, len(z)), replace=False)
    for i in idx:
        axrow[1].plot(x, z[i], "-", color=(GREEN if y[i] == 0 else RED),
                      alpha=0.06, lw=0.6)
    for t, c, lab in [(zc, GREEN, "correct"), (zi, RED, "incorrect")]:
        mu, ci = mean_ci(t)
        axrow[1].plot(x, mu, "-o", color=c, label=lab, lw=2.2)
        axrow[1].fill_between(x, mu - ci, mu + ci, color=c, alpha=0.25)
    axrow[1].set_title(f"{metric}: per-layer z-scored (shape) + {len(idx)} samples")
    axrow[1].set_xlabel("layer"); axrow[1].set_ylabel("z"); axrow[1].legend()

    # per-layer single-number discriminability
    stats = []
    for j, L in enumerate(layers):
        col = traj[:, j]
        stats.append({
            "layer": int(L),
            "auc": round(directed_auc(col, y), 4),
            "cohens_d": round(cohens_d(traj[y == 1, j], traj[y == 0, j]), 4),
            "mean_correct": float(tc[:, j].mean()),
            "mean_incorrect": float(ti[:, j].mean()),
        })
    return stats


def plot_subset(layers, traj_by_metric, y, rng, n_per_class, out_png, stem, token):
    """A readable figure: n_per_class individual trajectories per group, drawn solidly.
    One row per metric; left = raw values, right = per-layer z-scored (shape)."""
    x = np.asarray(layers)
    metrics = list(traj_by_metric)
    ic = rng.choice(np.where(y == 0)[0], size=min(n_per_class, int((y == 0).sum())),
                    replace=False)
    ii = rng.choice(np.where(y == 1)[0], size=min(n_per_class, int((y == 1).sum())),
                    replace=False)
    fig, axes = plt.subplots(len(metrics), 2,
                             figsize=(13, 4.6 * len(metrics)), squeeze=False)
    for r, metric in enumerate(metrics):
        traj = traj_by_metric[metric]
        mu_all = traj.mean(axis=0); sd_all = traj.std(axis=0) + 1e-9
        z = (traj - mu_all) / sd_all
        for col, (data, title, ylab) in enumerate(
                [(traj, f"{metric}: {len(ic)+len(ii)} individual steps (raw)", metric),
                 (z, f"{metric}: same steps, per-layer z-scored (shape)", "z")]):
            ax = axes[r, col]
            for i in ic:
                ax.plot(x, data[i], "-o", color=GREEN, alpha=0.55, lw=1.0, ms=3)
            for i in ii:
                ax.plot(x, data[i], "-o", color=RED, alpha=0.55, lw=1.0, ms=3)
            # group means on top, bold
            ax.plot(x, data[y == 0].mean(0), "-", color="#0a6b22", lw=3, label="correct mean")
            ax.plot(x, data[y == 1].mean(0), "-", color="#8a0d2c", lw=3, label="incorrect mean")
            ax.set_title(title); ax.set_xlabel("layer"); ax.set_ylabel(ylab); ax.legend()
    fig.suptitle(f"Layer trajectories (subset) - PRM800K {stem}  token={token}  "
                 f"green=correct red=incorrect", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"[subset] -> {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--stem", type=str, default="prm800k_heldout_test")
    ap.add_argument("--token", type=str, default="last", choices=["first", "last"])
    ap.add_argument("--metrics", nargs="+", default=["raw_max", "l2_norm"])
    ap.add_argument("--n_sample", type=int, default=400,
                    help="individual trajectories drawn faintly on the z panel")
    ap.add_argument("--subset_n", type=int, default=15,
                    help="readable individual trajectories per class in the subset figure")
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    manifest = json.loads(
        (args.merged_dir / f"{args.stem}_manifest.json").read_text())
    layers = list(manifest["layer_indices"])

    # one mmap slice per layer (token fixed); labels identical across layers
    d_by_layer = {L: load_prm800k_multitoken(args.merged_dir, args.stem, L, args.token)
                  for L in layers}
    y = d_by_layer[layers[0]].label.astype(int)       # 1 = incorrect
    n = len(y)
    floor = max(np.bincount(y)) / n
    print(f"[traj] stem={args.stem} token={args.token} n={n} layers={layers} "
          f"floor={floor:.3f} (incorrect={int(y.sum())})")

    R = {"stem": args.stem, "token": args.token, "n": n, "layers": layers,
         "floor": round(floor, 4), "metrics": {}}

    fig, axes = plt.subplots(len(args.metrics), 2,
                             figsize=(13, 4.6 * len(args.metrics)), squeeze=False)
    traj_by_metric = {}
    for r, metric in enumerate(args.metrics):
        traj = build_trajectories(d_by_layer, layers, metric)
        traj_by_metric[metric] = traj
        stats = plot_metric(axes[r], layers, traj, y, metric, rng, args.n_sample)
        R["metrics"][metric] = stats
        print(f"\n[{metric}] per-layer single-number discriminability:")
        for s in stats:
            print(f"  L{s['layer']:>2}  AUC={s['auc']:.3f}  d={s['cohens_d']:+.3f}  "
                  f"(corr {s['mean_correct']:.3g} vs incorr {s['mean_incorrect']:.3g})")

    fig.suptitle(f"Layer trajectories - PRM800K {args.stem}  token={args.token}  "
                 f"(n={n}, floor={floor:.2f}, 1=incorrect)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = args.out_dir / f"trajectory_{args.token}.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    (args.out_dir / f"trajectory_{args.token}.json").write_text(json.dumps(R, indent=2))

    plot_subset(layers, traj_by_metric, y, np.random.default_rng(SEED), args.subset_n,
                args.out_dir / f"trajectory_subset_{args.token}.png", args.stem, args.token)
    print(f"\n[done] -> {png}")
    print(f"        + {args.out_dir / f'trajectory_{args.token}.json'}")


if __name__ == "__main__":
    main()
