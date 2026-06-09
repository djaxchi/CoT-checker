#!/usr/bin/env python3
"""Visualize correct-vs-incorrect representation geometry and probe separability
across the model-size DenseLinear ablation.

For each backbone it uses the saved PRM800K val_1k hidden states (500 correct /
500 incorrect, balanced) + the trained linear probe, and emits three figures:

  1. embeddings_<src>.png   2D projection (PCA, and t-SNE if --tsne) of the
                            hidden states, one panel per size, colored by class.
                            Shows how the geometry of correct vs first-error
                            steps changes with scale.
  2. probe_scores_<src>.png per-size distribution of the probe's P(error) for
                            each class, with the deployed val-selected threshold
                            marked. Shows the 1D separability the probe uses and
                            why the val threshold succeeds or collapses.
  3. separability_<src>.png summary curves vs model size: probe AUROC (geometry-
                            independent class separability) alongside oracle and
                            val-selected macro F1_PB.

Label convention (this repo): y=0 correct (rating +1), y=1 incorrect/non-viable
(rating -1); the probe outputs P(y=1)=P(error).

Run (TamIA login node is fine; CPU only):
    python scripts/s1ms_viz_representations.py --runs_root runs/s1_model_size_dense [--tsne]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_TAGS = ["qwen2_5_1_5b", "qwen2_5_3b", "qwen2_5_7b", "qwen2_5_14b", "qwen2_5_32b"]
CORRECT_COLOR = "#2c7fb8"   # y=0 correct
ERROR_COLOR = "#e6550d"     # y=1 incorrect / first-error


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pca2(h: np.ndarray) -> np.ndarray:
    """Top-2 PCA scores via numpy SVD (no sklearn dependency)."""
    hc = h - h.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(hc, full_matrices=False)
    return u[:, :2] * s[:2]


def auroc_np(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC for positive class y==1, tie-aware (average ranks). No sklearn."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks_sorted = np.empty(len(s), dtype=float)
    j = 0
    while j < len(s):
        k = j
        while k + 1 < len(s) and s_sorted[k + 1] == s_sorted[j]:
            k += 1
        ranks_sorted[j:k + 1] = (j + 1 + k + 1) / 2.0
        j = k + 1
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = ranks_sorted
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_probe_wb(pt_path: Path) -> tuple[np.ndarray, float]:
    """Load the linear probe state_dict and return (w, b) as numpy."""
    import torch
    sd = torch.load(pt_path, map_location="cpu", weights_only=False)
    w = sd["fc.weight"].detach().cpu().numpy().reshape(-1)
    b = float(sd["fc.bias"].detach().cpu().numpy().reshape(-1)[0])
    return w, b


def load_model(model_dir: Path) -> dict | None:
    h_path = model_dir / "merged" / "val_1k_h.npy"
    y_path = model_dir / "merged" / "val_1k_y.npy"
    probe = model_dir / "linear_probe.pt"
    if not (h_path.exists() and y_path.exists() and probe.exists()):
        return None
    h = np.load(h_path).astype(np.float32)
    y = np.load(y_path).astype(int)
    w, b = load_probe_wb(probe)
    thr_json = model_dir / "threshold.json"
    thr = json.loads(thr_json.read_text())["selected_threshold"] if thr_json.exists() else 0.5
    cfg = {}
    if (model_dir / "model_config.json").exists():
        cfg = json.loads((model_dir / "model_config.json").read_text())
    psm = {}
    if (model_dir / "per_subset_metrics.json").exists():
        psm = json.loads((model_dir / "per_subset_metrics.json").read_text())

    logits = h @ w + b
    scores = sigmoid(logits)
    auroc = auroc_np(y, scores)
    return {
        "h": h, "y": y, "scores": scores, "logits": logits,
        "thr": float(thr), "auroc": auroc,
        "label": cfg.get("params_label", model_dir.name),
        "hidden_size": cfg.get("hidden_size"),
        "val_macro": psm.get("macro_f1_val_threshold"),
        "oracle_macro": psm.get("macro_f1_oracle"),
    }


def _scatter_2d(ax, xy, y, title):
    for cls, color, name in [(0, CORRECT_COLOR, "correct"), (1, ERROR_COLOR, "incorrect")]:
        m = y == cls
        ax.scatter(xy[m, 0], xy[m, 1], s=6, c=color, alpha=0.5, label=name, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def fig_embeddings(models: list[dict], out: Path, use_tsne: bool) -> None:
    methods = ["PCA"] + (["t-SNE"] if use_tsne else [])
    n = len(models)
    fig, axes = plt.subplots(len(methods), n, figsize=(3.1 * n, 3.1 * len(methods)), squeeze=False)
    for r, method in enumerate(methods):
        for c, m in enumerate(models):
            h = m["h"]
            if method == "PCA":
                xy = pca2(h)
            else:
                from sklearn.manifold import TSNE  # lazy: only needed for --tsne
                perp = min(30, max(5, (len(h) - 1) // 3))
                xy = TSNE(n_components=2, init="pca", random_state=42,
                          perplexity=perp).fit_transform(h)
            _scatter_2d(axes[r, c], xy, m["y"],
                        f"{m['label']}  ({method})\nAUROC={m['auroc']:.3f}")
            if c == 0:
                axes[r, c].set_ylabel(method, fontsize=11)
    axes[0, 0].legend(loc="upper right", fontsize=8, markerscale=2, framealpha=0.9)
    fig.suptitle("Correct vs incorrect step geometry (PRM800K val_1k) across model size", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {out}")


def fig_probe_scores(models: list[dict], out: Path) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    bins = np.linspace(0, 1, 41)
    for c, m in enumerate(models):
        ax = axes[0, c]
        s, y = m["scores"], m["y"]
        ax.hist(s[y == 0], bins=bins, color=CORRECT_COLOR, alpha=0.6, label="correct", density=True)
        ax.hist(s[y == 1], bins=bins, color=ERROR_COLOR, alpha=0.6, label="incorrect", density=True)
        ax.axvline(m["thr"], color="k", ls="--", lw=1.5, label=f"val t={m['thr']:.2f}")
        vmac = m["val_macro"]
        ax.set_title(f"{m['label']}\nAUROC={m['auroc']:.3f}  "
                     f"valF1={vmac:.3f}" if vmac is not None else f"{m['label']}\nAUROC={m['auroc']:.3f}",
                     fontsize=10)
        ax.set_xlabel("P(error)")
        ax.set_xlim(0, 1)
        if c == 0:
            ax.set_ylabel("density"); ax.legend(fontsize=8)
    fig.suptitle("Probe score distributions by class (PRM800K val_1k); dashed = deployed threshold",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {out}")


def fig_separability(models: list[dict], out: Path) -> None:
    xs = list(range(len(models)))
    labels = [m["label"] for m in models]
    auroc = [m["auroc"] for m in models]
    oracle = [m["oracle_macro"] for m in models]
    valm = [m["val_macro"] for m in models]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(xs, auroc, "o-", color="#225522", label="probe AUROC (val_1k separability)")
    ax1.set_ylabel("AUROC", color="#225522")
    ax1.set_xticks(xs); ax1.set_xticklabels(labels)
    ax1.set_xlabel("model size")
    ax1.tick_params(axis="y", labelcolor="#225522")

    ax2 = ax1.twinx()
    if all(v is not None for v in oracle):
        ax2.plot(xs, oracle, "s--", color="#7a0177", label="oracle macro F1_PB")
    if all(v is not None for v in valm):
        ax2.plot(xs, valm, "^:", color="#c51b8a", label="val-selected macro F1_PB")
    ax2.set_ylabel("macro F1_PB")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="lower right")
    ax1.set_title("Class separability and ProcessBench F1_PB vs model size")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, default=Path("runs/s1_model_size_dense"))
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Default: <runs_root>/figures")
    p.add_argument("--tsne", action="store_true",
                   help="Also compute a t-SNE row (slower; clearer clusters).")
    args = p.parse_args()

    out_dir = args.out_dir or (args.runs_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    models: list[dict] = []
    for tag in args.tags:
        m = load_model(args.runs_root / tag)
        if m is None:
            print(f"[viz] skip {tag}: missing val_1k encodings or probe")
            continue
        models.append(m)
        print(f"[viz] {m['label']}: n={len(m['y'])} "
              f"(corr={int((m['y']==0).sum())}/inc={int((m['y']==1).sum())}) "
              f"AUROC={m['auroc']:.3f} val_t={m['thr']:.2f} "
              f"oracle={m['oracle_macro']} val={m['val_macro']}")
    if not models:
        raise SystemExit(f"[viz] no model data under {args.runs_root}")

    fig_embeddings(models, out_dir / "embeddings_val1k.png", args.tsne)
    fig_probe_scores(models, out_dir / "probe_scores_val1k.png")
    fig_separability(models, out_dir / "separability_val1k.png")
    print(f"[viz] done -> {out_dir}")


if __name__ == "__main__":
    main()
