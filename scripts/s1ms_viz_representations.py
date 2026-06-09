#!/usr/bin/env python3
"""Visualize correct-vs-incorrect representation geometry and probe separability
across the model-size DenseLinear ablation.

Two data sources:

  --source val  (default)  PRM800K val_1k hidden states (500 correct / 500
                           incorrect, balanced). Clean class-geometry picture.
  --source pb              ProcessBench step hidden states per subset. This is
                           the deployment distribution, where olympiadbench
                           failure lives. Per-step class follows the metric:
                             positive = the first-error step (step_idx == label)
                             negative = correct-context steps (correct traces +
                                        pre-error steps; step_idx < label)
                             dropped  = post-error steps (step_idx > label), ambiguous

Figures (written to <runs_root>/figures/):
  val source:
    embeddings_val1k.png      2D PCA (and t-SNE if --tsne) per size, by class
    probe_scores_val1k.png    per-size P(error) distribution by class + threshold
    separability_val1k.png    AUROC + oracle/val macro F1_PB vs size
  pb source:
    probe_scores_pb.png       subset x size grid of P(error) distributions by
                              class + the deployed threshold (the key per-subset
                              success/failure view)
    embeddings_pb_<subset>.png  2D projection per size for one subset, by class
    separability_pb.png       per-subset AUROC vs size + per-subset F1_PB

Label convention: positive class = "incorrect / first-error"; the probe outputs
P(error). Run on TamIA login node (CPU); core needs only numpy/torch/matplotlib
(sklearn is imported lazily only for --tsne).
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
SUBSETS = ["gsm8k", "math", "olympiadbench", "omnimath"]
CORRECT_COLOR = "#2c7fb8"   # negative: correct / correct-context
ERROR_COLOR = "#e6550d"     # positive: incorrect / first-error
EMBED_CAP = 2500            # max points plotted per panel (subsample for speed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


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
    import torch
    sd = torch.load(pt_path, map_location="cpu", weights_only=False)
    w = sd["fc.weight"].detach().cpu().numpy().reshape(-1).astype(np.float64)
    b = float(sd["fc.bias"].detach().cpu().numpy().reshape(-1)[0])
    return w, b


def _read_meta(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _base(model_dir: Path) -> dict:
    """Shared per-model metadata (probe, threshold, config, macros)."""
    w, b = load_probe_wb(model_dir / "linear_probe.pt")
    thr_json = model_dir / "threshold.json"
    thr = json.loads(thr_json.read_text())["selected_threshold"] if thr_json.exists() else 0.5
    cfg = json.loads((model_dir / "model_config.json").read_text()) if (model_dir / "model_config.json").exists() else {}
    psm = json.loads((model_dir / "per_subset_metrics.json").read_text()) if (model_dir / "per_subset_metrics.json").exists() else {}
    return {"w": w, "b": b, "thr": float(thr),
            "label": cfg.get("params_label", model_dir.name),
            "hidden_size": cfg.get("hidden_size"),
            "val_macro": psm.get("macro_f1_val_threshold"),
            "oracle_macro": psm.get("macro_f1_oracle"),
            "per_subset": psm.get("per_subset", {})}


def load_val(model_dir: Path) -> dict | None:
    h_path = model_dir / "merged" / "val_1k_h.npy"
    y_path = model_dir / "merged" / "val_1k_y.npy"
    if not (h_path.exists() and y_path.exists() and (model_dir / "linear_probe.pt").exists()):
        return None
    m = _base(model_dir)
    h = np.load(h_path).astype(np.float64)
    y = np.load(y_path).astype(int)
    scores = sigmoid(h @ m["w"] + m["b"])
    m.update({"h": h, "y": y, "scores": scores, "auroc": auroc_np(y, scores)})
    return m


def load_pb_subset(model_dir: Path, subset: str, w: np.ndarray, b: float) -> dict | None:
    out = model_dir / "processbench_eval_shards" / subset
    h_path, m_path = out / "pb_step_h.npy", out / "pb_step_meta.jsonl"
    if not (h_path.exists() and m_path.exists()):
        return None
    h = np.load(h_path).astype(np.float64)
    meta = _read_meta(m_path)
    if len(meta) != h.shape[0]:
        return None
    y = np.full(len(meta), -1, dtype=int)   # -1 = drop (post-error)
    for i, r in enumerate(meta):
        lab, si = int(r["label"]), int(r["step_idx"])
        if lab == -1 or si < lab:
            y[i] = 0                         # correct-context
        elif si == lab:
            y[i] = 1                         # first error
        # else leave -1 (post-error, ambiguous)
    keep = y >= 0
    h, y = h[keep], y[keep]
    scores = sigmoid(h @ w + b)
    return {"h": h, "y": y, "scores": scores, "auroc": auroc_np(y, scores)}


def load_pb(model_dir: Path, subsets: list[str]) -> dict | None:
    if not (model_dir / "linear_probe.pt").exists():
        return None
    m = _base(model_dir)
    m["subsets"] = {}
    for s in subsets:
        d = load_pb_subset(model_dir, s, m["w"], m["b"])
        if d is not None:
            m["subsets"][s] = d
    return m if m["subsets"] else None


def _subsample(h, y, cap, seed=42):
    if len(y) <= cap:
        return h, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=cap, replace=False)
    return h[idx], y[idx]


def _scatter_2d(ax, xy, y, title):
    for cls, color, name in [(0, CORRECT_COLOR, "correct"), (1, ERROR_COLOR, "first-error")]:
        msk = y == cls
        ax.scatter(xy[msk, 0], xy[msk, 1], s=6, c=color, alpha=0.5, label=name, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _pca_k(h: np.ndarray, k: int) -> np.ndarray:
    hc = h - h.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(hc, full_matrices=False)
    k = min(k, u.shape[1])
    return u[:, :k] * s[:k]


def _project(h, method):
    if method == "PCA":
        return pca2(h)
    from sklearn.manifold import TSNE  # lazy
    # Standard practice: PCA-reduce to ~50 dims before t-SNE. Much faster than
    # running it on the full hidden dim (1536-5120), and denoises the input.
    hp = _pca_k(h, 50)
    perp = min(30, max(5, (len(h) - 1) // 3))
    return TSNE(n_components=2, init="pca", random_state=42, perplexity=perp).fit_transform(hp)


def _score_panel(ax, scores, y, thr, title):
    bins = np.linspace(0, 1, 41)
    ax.hist(scores[y == 0], bins=bins, color=CORRECT_COLOR, alpha=0.6, density=True, label="correct")
    ax.hist(scores[y == 1], bins=bins, color=ERROR_COLOR, alpha=0.6, density=True, label="first-error")
    ax.axvline(thr, color="k", ls="--", lw=1.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1)


# --------------------------------------------------------------------------- val figures

def fig_val(models, out_dir, use_tsne):
    methods = ["PCA"] + (["t-SNE"] if use_tsne else [])
    n = len(models)
    fig, axes = plt.subplots(len(methods), n, figsize=(3.1 * n, 3.1 * len(methods)), squeeze=False)
    for r, method in enumerate(methods):
        for c, m in enumerate(models):
            h, y = _subsample(m["h"], m["y"], EMBED_CAP)
            _scatter_2d(axes[r, c], _project(h, method), y, f"{m['label']} ({method})\nAUROC={m['auroc']:.3f}")
            if c == 0:
                axes[r, c].set_ylabel(method)
    axes[0, 0].legend(loc="upper right", fontsize=8, markerscale=2)
    fig.suptitle("Correct vs incorrect geometry (PRM800K val_1k) across model size")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "embeddings_val1k.png", dpi=150); plt.close(fig)
    print(f"[viz] wrote {out_dir/'embeddings_val1k.png'}")

    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4), squeeze=False)
    for c, m in enumerate(models):
        t = f"{m['label']}\nAUROC={m['auroc']:.3f}"
        if m["val_macro"] is not None:
            t += f"  valF1={m['val_macro']:.3f}"
        _score_panel(axes[0, c], m["scores"], m["y"], m["thr"], t)
        axes[0, c].set_xlabel("P(error)")
        if c == 0:
            axes[0, c].set_ylabel("density"); axes[0, c].legend(fontsize=8)
    fig.suptitle("Probe score distributions by class (val_1k); dashed = deployed threshold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "probe_scores_val1k.png", dpi=150); plt.close(fig)
    print(f"[viz] wrote {out_dir/'probe_scores_val1k.png'}")

    _fig_sep_single(models, out_dir / "separability_val1k.png")


def _fig_sep_single(models, out):
    xs = list(range(len(models)))
    labels = [m["label"] for m in models]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(xs, [m["auroc"] for m in models], "o-", color="#225522", label="probe AUROC")
    ax1.set_ylabel("AUROC", color="#225522"); ax1.set_xticks(xs); ax1.set_xticklabels(labels)
    ax1.set_xlabel("model size"); ax1.tick_params(axis="y", labelcolor="#225522")
    ax2 = ax1.twinx()
    if all(m["oracle_macro"] is not None for m in models):
        ax2.plot(xs, [m["oracle_macro"] for m in models], "s--", color="#7a0177", label="oracle macro F1_PB")
    if all(m["val_macro"] is not None for m in models):
        ax2.plot(xs, [m["val_macro"] for m in models], "^:", color="#c51b8a", label="val-selected macro F1_PB")
    ax2.set_ylabel("macro F1_PB")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax1.set_title("Class separability and ProcessBench F1_PB vs model size")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[viz] wrote {out}")


# --------------------------------------------------------------------------- pb figures

def fig_pb_scores_grid(models, subsets, out):
    """subset (rows) x size (cols) grid of P(error) distributions by class."""
    nr, nc = len(subsets), len(models)
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.5 * nr), squeeze=False)
    for r, sub in enumerate(subsets):
        for c, m in enumerate(models):
            ax = axes[r, c]
            d = m["subsets"].get(sub)
            if d is None:
                ax.set_axis_off(); continue
            ps = m["per_subset"].get(sub, {})
            valf1 = ps.get("val_selected", {}).get("F1_PB")
            orf1 = ps.get("oracle", {}).get("F1_PB")
            t = f"{m['label']} / {sub}\nAUROC={d['auroc']:.3f}"
            if valf1 is not None and orf1 is not None:
                t += f"\nvalF1={valf1:.2f} orF1={orf1:.2f}"
            _score_panel(ax, d["scores"], d["y"], m["thr"], t)
            if r == nr - 1:
                ax.set_xlabel("P(error)")
            if c == 0:
                ax.set_ylabel(sub, fontsize=10)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("ProcessBench probe scores by class (rows=subset, cols=size); dashed=deployed threshold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[viz] wrote {out}")


def fig_pb_embeddings(models, subset, out, use_tsne):
    methods = ["PCA"] + (["t-SNE"] if use_tsne else [])
    n = len(models)
    fig, axes = plt.subplots(len(methods), n, figsize=(3.1 * n, 3.1 * len(methods)), squeeze=False)
    for r, method in enumerate(methods):
        for c, m in enumerate(models):
            ax = axes[r, c]
            d = m["subsets"].get(subset)
            if d is None:
                ax.set_axis_off(); continue
            h, y = _subsample(d["h"], d["y"], EMBED_CAP)
            _scatter_2d(ax, _project(h, method), y, f"{m['label']} ({method})\nAUROC={d['auroc']:.3f}")
            if c == 0:
                ax.set_ylabel(method)
    axes[0, 0].legend(loc="upper right", fontsize=8, markerscale=2)
    fig.suptitle(f"ProcessBench {subset}: correct vs first-error geometry across model size")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[viz] wrote {out}")


def fig_pb_separability(models, subsets, out):
    xs = list(range(len(models)))
    labels = [m["label"] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    cmap = plt.get_cmap("viridis")
    for i, sub in enumerate(subsets):
        col = cmap(i / max(len(subsets) - 1, 1))
        auroc = [m["subsets"].get(sub, {}).get("auroc", np.nan) for m in models]
        orf1 = [m["per_subset"].get(sub, {}).get("oracle", {}).get("F1_PB", np.nan) for m in models]
        axes[0].plot(xs, auroc, "o-", color=col, label=sub)
        axes[1].plot(xs, orf1, "s--", color=col, label=sub)
    axes[0].set_title("Probe AUROC (first-error vs correct) vs size")
    axes[0].set_ylabel("AUROC")
    axes[1].set_title("Per-subset oracle F1_PB vs size")
    axes[1].set_ylabel("oracle F1_PB")
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_xlabel("model size"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[viz] wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, default=Path("runs/s1_model_size_dense"))
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    p.add_argument("--source", choices=["val", "pb"], default="val")
    p.add_argument("--subsets", nargs="+", default=SUBSETS, help="ProcessBench subsets (--source pb).")
    p.add_argument("--pb_embed_subset", default="olympiadbench",
                   help="Which subset to render the 2D embedding for (--source pb).")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--tsne", action="store_true", help="Also compute a t-SNE row (slower; needs sklearn).")
    args = p.parse_args()

    out_dir = args.out_dir or (args.runs_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for tag in args.tags:
        md = args.runs_root / tag
        m = load_val(md) if args.source == "val" else load_pb(md, args.subsets)
        if m is None:
            print(f"[viz] skip {tag}: missing {args.source} data / probe")
            continue
        models.append(m)
        if args.source == "val":
            print(f"[viz] {m['label']}: val_1k AUROC={m['auroc']:.3f} t={m['thr']:.2f}")
        else:
            subs = ", ".join(f"{s}:AUROC={m['subsets'][s]['auroc']:.3f}" for s in m["subsets"])
            print(f"[viz] {m['label']}: t={m['thr']:.2f}  {subs}")
    if not models:
        raise SystemExit(f"[viz] no {args.source} data under {args.runs_root}")

    if args.source == "val":
        fig_val(models, out_dir, args.tsne)
    else:
        present = [s for s in args.subsets if any(s in m["subsets"] for m in models)]
        fig_pb_scores_grid(models, present, out_dir / "probe_scores_pb.png")
        if any(args.pb_embed_subset in m["subsets"] for m in models):
            fig_pb_embeddings(models, args.pb_embed_subset,
                              out_dir / f"embeddings_pb_{args.pb_embed_subset}.png", args.tsne)
        fig_pb_separability(models, present, out_dir / "separability_pb.png")
    print(f"[viz] done -> {out_dir}")


if __name__ == "__main__":
    main()
