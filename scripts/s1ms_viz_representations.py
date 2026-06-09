#!/usr/bin/env python3
"""Visualize correct-vs-incorrect representation geometry across the model-size
DenseLinear ablation. Metrics shown are F1 only (PRM800K val step F1 and
ProcessBench macro F1_PB), so figures stay directly comparable to the leaderboard
and to every other result in this project.

Sources:

  --source val    PRM800K val_1k hidden states (500 correct / 500 incorrect).
  --source pb     ProcessBench step hidden states per subset (deployment dist).
                  Per-step class follows the metric: positive = first-error step
                  (step_idx == label); negative = correct-context (correct traces
                  + pre-error steps); post-error steps dropped.
  --source forks  MATCHED PRM800K fork siblings: same prefix, a correct (positive)
                  and an incorrect (negative) next step. This controls for content,
                  so the geometric difference between the two points is purely the
                  correctness signal. Needs per-size fork encodings under
                  <runs_root>/<tag>/forks/ (run slurm/s1_model_size/run_fork_encode_sweep.sh).

Figures land in <runs_root>/figures/. Metric shown everywhere = macro F1_PB
(val-selected + oracle), from per_subset_metrics.json.
"""

from __future__ import annotations

import os
# Cap BLAS threads BEFORE importing numpy: on a shared login node numpy/OpenBLAS
# otherwise spawns one thread per core (64+) and thrashes on tiny gemm/eigh ops,
# stalling for minutes. Override by exporting these before running.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_TAGS = ["qwen2_5_1_5b", "qwen2_5_3b", "qwen2_5_7b", "qwen2_5_14b", "qwen2_5_32b"]
SUBSETS = ["gsm8k", "math", "olympiadbench", "omnimath"]
CORRECT_COLOR = "#2c7fb8"   # correct / correct-context / positive sibling
ERROR_COLOR = "#e6550d"     # incorrect / first-error / negative sibling
EMBED_CAP = 2500


def _log(msg: str) -> None:
    print(f"[viz {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def pca_scores(h: np.ndarray, k: int) -> np.ndarray:
    """Top-k PCA scores via the Gram matrix (fast for n < d)."""
    hc = (h - h.mean(axis=0, keepdims=True)).astype(np.float32)
    gram = hc @ hc.T
    w, v = np.linalg.eigh(gram)
    k = min(k, v.shape[1])
    w = np.clip(w[::-1][:k], 0.0, None)
    v = v[:, ::-1][:, :k]
    return v * np.sqrt(w)[None, :]


def step_f1_macro(scores: np.ndarray, y: np.ndarray, thr: float) -> float:
    """Macro F1 over the two step classes (correct=0, incorrect=1) at threshold
    thr, on PRM800K val. This is an F1 on the val data itself (the probe's
    actual classification), distinct from the trace-level ProcessBench F1_PB."""
    pred = (scores > thr).astype(int)
    f1s = []
    for cls in (0, 1):
        tp = int(((pred == cls) & (y == cls)).sum())
        fp = int(((pred == cls) & (y != cls)).sum())
        fn = int(((pred != cls) & (y == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / 2.0


def f1_str(m: dict) -> str:
    v, o = m.get("val_macro"), m.get("oracle_macro")
    parts = []
    if v is not None:
        parts.append(f"valF1={v:.3f}")
    if o is not None:
        parts.append(f"orF1={o:.3f}")
    return "  ".join(parts)


def load_probe_wb(pt_path: Path) -> tuple[np.ndarray, float]:
    import torch
    sd = torch.load(pt_path, map_location="cpu", weights_only=False)
    w = sd["fc.weight"].detach().cpu().numpy().reshape(-1).astype(np.float64)
    b = float(sd["fc.bias"].detach().cpu().numpy().reshape(-1)[0])
    return w, b


def _read_meta(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _base(model_dir: Path) -> dict:
    w, b = load_probe_wb(model_dir / "linear_probe.pt")
    thr_json = model_dir / "threshold.json"
    thr = json.loads(thr_json.read_text())["selected_threshold"] if thr_json.exists() else 0.5
    cfg = json.loads((model_dir / "model_config.json").read_text()) if (model_dir / "model_config.json").exists() else {}
    psm = json.loads((model_dir / "per_subset_metrics.json").read_text()) if (model_dir / "per_subset_metrics.json").exists() else {}
    return {"w": w, "b": b, "thr": float(thr),
            "label": cfg.get("params_label", model_dir.name),
            "val_macro": psm.get("macro_f1_val_threshold"),
            "oracle_macro": psm.get("macro_f1_oracle"),
            "per_subset": psm.get("per_subset", {})}


def load_val(model_dir: Path) -> dict | None:
    h_path, y_path = model_dir / "merged" / "val_1k_h.npy", model_dir / "merged" / "val_1k_y.npy"
    if not (h_path.exists() and y_path.exists() and (model_dir / "linear_probe.pt").exists()):
        return None
    m = _base(model_dir)
    h = np.load(h_path).astype(np.float32)
    y = np.load(y_path).astype(int)
    scores = sigmoid(h @ m["w"] + m["b"])
    m.update({"h": h, "y": y, "scores": scores,
              "val_step_f1": step_f1_macro(scores, y, m["thr"])})
    return m


def load_pb_subset(model_dir: Path, subset: str, w, b) -> dict | None:
    out = model_dir / "processbench_eval_shards" / subset
    h_path, m_path = out / "pb_step_h.npy", out / "pb_step_meta.jsonl"
    if not (h_path.exists() and m_path.exists()):
        return None
    h = np.load(h_path).astype(np.float32)
    meta = _read_meta(m_path)
    if len(meta) != h.shape[0]:
        return None
    y = np.full(len(meta), -1, dtype=int)
    for i, r in enumerate(meta):
        lab, si = int(r["label"]), int(r["step_idx"])
        if lab == -1 or si < lab:
            y[i] = 0
        elif si == lab:
            y[i] = 1
    keep = y >= 0
    h, y = h[keep], y[keep]
    return {"h": h, "y": y, "scores": sigmoid(h @ w + b)}


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


def load_forks(model_dir: Path, stem: str, max_forks: int | None) -> dict | None:
    fdir = model_dir / "forks"
    h_path, m_path = fdir / f"{stem}_h.npy", fdir / f"{stem}_meta.jsonl"
    if not (h_path.exists() and m_path.exists() and (model_dir / "linear_probe.pt").exists()):
        return None
    m = _base(model_dir)
    h = np.load(h_path).astype(np.float32)
    meta = _read_meta(m_path)
    by_fork: dict = defaultdict(dict)
    for r in meta:
        by_fork[r["fork_id"]].setdefault(r["role"], int(r["row"]))
    pos_i, neg_i, anc_i = [], [], []
    for fid, roles in by_fork.items():
        if "positive" in roles and "negative" in roles:
            pos_i.append(roles["positive"]); neg_i.append(roles["negative"])
            anc_i.append(roles.get("anchor", roles["positive"]))
    if not pos_i:
        return None
    if max_forks is not None and len(pos_i) > max_forks:
        sel = np.random.default_rng(42).choice(len(pos_i), size=max_forks, replace=False)
        pos_i = [pos_i[k] for k in sel]; neg_i = [neg_i[k] for k in sel]; anc_i = [anc_i[k] for k in sel]
    m.update({"pos_h": h[pos_i], "neg_h": h[neg_i], "anc_h": h[anc_i], "n_forks": len(pos_i)})
    return m


def _subsample(h, y, cap, seed=42):
    if len(y) <= cap:
        return h, y
    idx = np.random.default_rng(seed).choice(len(y), size=cap, replace=False)
    return h[idx], y[idx]


def _scatter(ax, xy, y, title, names=("correct", "incorrect")):
    for cls, color, name in [(0, CORRECT_COLOR, names[0]), (1, ERROR_COLOR, names[1])]:
        msk = y == cls
        ax.scatter(xy[msk, 0], xy[msk, 1], s=6, c=color, alpha=0.5, label=name, linewidths=0)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def _project(h, method, tag=""):
    t0 = time.perf_counter()
    if method == "PCA":
        xy = pca_scores(h, 2)
    else:
        from sklearn.manifold import TSNE
        hp = pca_scores(h, 50)
        perp = min(30, max(5, (len(h) - 1) // 3))
        xy = TSNE(n_components=2, init="pca", random_state=42, perplexity=perp).fit_transform(hp)
    _log(f"  projected {tag} {method} (n={len(h)}, d={h.shape[1]}) in {time.perf_counter()-t0:.1f}s")
    return xy


# --------------------------------------------------------------------------- val / pb shared

def _score_panel(ax, scores, y, thr, title, names):
    bins = np.linspace(0, 1, 41)
    ax.hist(scores[y == 0], bins=bins, color=CORRECT_COLOR, alpha=0.6, density=True, label=names[0])
    ax.hist(scores[y == 1], bins=bins, color=ERROR_COLOR, alpha=0.6, density=True, label=names[1])
    ax.axvline(thr, color="k", ls="--", lw=1.4)
    ax.set_title(title, fontsize=9); ax.set_xlim(0, 1)


def fig_val(models, out_dir, use_tsne):
    methods = ["PCA"] + (["t-SNE"] if use_tsne else [])
    n = len(models)
    _log("building val embeddings figure ...")
    fig, axes = plt.subplots(len(methods), n, figsize=(3.1 * n, 3.1 * len(methods)), squeeze=False)
    for r, method in enumerate(methods):
        for c, m in enumerate(models):
            h, y = _subsample(m["h"], m["y"], EMBED_CAP)
            _scatter(axes[r, c], _project(h, method, m["label"]), y,
                     f"{m['label']} ({method})\nval F1={m['val_step_f1']:.3f}")
            if c == 0:
                axes[r, c].set_ylabel(method)
    axes[0, 0].legend(loc="upper right", fontsize=8, markerscale=2)
    fig.suptitle("Correct vs incorrect geometry (PRM800K val_1k); val F1 = probe step F1 at deployed threshold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out_dir / "embeddings_val1k.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'embeddings_val1k.png'}")

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    for c, m in enumerate(models):
        _score_panel(axes[0, c], m["scores"], m["y"], m["thr"],
                     f"{m['label']}\nval F1={m['val_step_f1']:.3f}", ("correct", "incorrect"))
        axes[0, c].set_xlabel("P(error)")
        if c == 0:
            axes[0, c].set_ylabel("density"); axes[0, c].legend(fontsize=8)
    fig.suptitle("Probe score by class (val_1k); dashed = deployed threshold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out_dir / "probe_scores_val1k.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'probe_scores_val1k.png'}")

    xs = list(range(n)); labels = [m["label"] for m in models]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, [m["val_step_f1"] for m in models], "o-", color="#225522",
            label="PRM800K val step F1 (probe @ threshold)")
    if all(m["oracle_macro"] is not None for m in models):
        ax.plot(xs, [m["oracle_macro"] for m in models], "s-", color="#7a0177", label="oracle macro F1_PB")
    if all(m["val_macro"] is not None for m in models):
        ax.plot(xs, [m["val_macro"] for m in models], "^--", color="#c51b8a", label="val-selected macro F1_PB")
    ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_xlabel("model size"); ax.set_ylabel("F1")
    ax.set_title("F1 vs model size (PRM800K val step F1 and ProcessBench macro F1_PB)"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "f1_vs_size.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'f1_vs_size.png'}")


def fig_pb(models, subsets, out_dir, embed_subset, use_tsne):
    present = [s for s in subsets if any(s in m["subsets"] for m in models)]
    nr, nc = len(present), len(models)
    _log("building pb score grid ...")
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.5 * nr), squeeze=False)
    for r, sub in enumerate(present):
        for c, m in enumerate(models):
            ax = axes[r, c]; d = m["subsets"].get(sub)
            if d is None:
                ax.set_axis_off(); continue
            ps = m["per_subset"].get(sub, {})
            vf, of = ps.get("val_selected", {}).get("F1_PB"), ps.get("oracle", {}).get("F1_PB")
            t = f"{m['label']} / {sub}"
            if vf is not None and of is not None:
                t += f"\nvalF1={vf:.2f} orF1={of:.2f}"
            _score_panel(ax, d["scores"], d["y"], m["thr"], t, ("correct", "first-error"))
            if r == nr - 1:
                ax.set_xlabel("P(error)")
            if c == 0:
                ax.set_ylabel(sub, fontsize=10)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("ProcessBench probe scores by class (rows=subset, cols=size); dashed=deployed threshold")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out_dir / "probe_scores_pb.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'probe_scores_pb.png'}")

    if any(embed_subset in m["subsets"] for m in models):
        methods = ["PCA"] + (["t-SNE"] if use_tsne else [])
        fig, axes = plt.subplots(len(methods), nc, figsize=(3.1 * nc, 3.1 * len(methods)), squeeze=False)
        for rr, method in enumerate(methods):
            for c, m in enumerate(models):
                ax = axes[rr, c]; d = m["subsets"].get(embed_subset)
                if d is None:
                    ax.set_axis_off(); continue
                h, y = _subsample(d["h"], d["y"], EMBED_CAP)
                _scatter(ax, _project(h, method, f"{m['label']}/{embed_subset}"), y,
                         f"{m['label']} ({method})", ("correct", "first-error"))
                if c == 0:
                    ax.set_ylabel(method)
        axes[0, 0].legend(loc="upper right", fontsize=8, markerscale=2)
        fig.suptitle(f"ProcessBench {embed_subset}: correct vs first-error geometry across size")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"embeddings_pb_{embed_subset}.png", dpi=150); plt.close(fig)
        _log(f"wrote {out_dir/('embeddings_pb_'+embed_subset+'.png')}")

    xs = list(range(nc)); labels = [m["label"] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    cmap = plt.get_cmap("viridis")
    for i, sub in enumerate(present):
        col = cmap(i / max(len(present) - 1, 1))
        vf = [m["per_subset"].get(sub, {}).get("val_selected", {}).get("F1_PB", np.nan) for m in models]
        of = [m["per_subset"].get(sub, {}).get("oracle", {}).get("F1_PB", np.nan) for m in models]
        axes[0].plot(xs, vf, "^--", color=col, label=sub)
        axes[1].plot(xs, of, "s-", color=col, label=sub)
    axes[0].set_title("Per-subset val-selected F1_PB vs size")
    axes[1].set_title("Per-subset oracle F1_PB vs size")
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_xlabel("model size")
        ax.set_ylabel("F1_PB"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "f1_vs_size_pb.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'f1_vs_size_pb.png'}")


# --------------------------------------------------------------------------- forks

def fig_forks(models, out_dir, use_tsne, max_lines):
    methods = ["PCA"] + (["t-SNE"] if use_tsne else [])
    n = len(models)
    _log("building forks embeddings figure ...")
    fig, axes = plt.subplots(len(methods), n, figsize=(3.3 * n, 3.3 * len(methods)), squeeze=False)
    for r, method in enumerate(methods):
        for c, m in enumerate(models):
            ax = axes[r, c]
            stacked = np.concatenate([m["pos_h"], m["neg_h"]], axis=0)
            xy = _project(stacked, method, f"{m['label']}/forks")
            k = len(m["pos_h"]); pos_xy, neg_xy = xy[:k], xy[k:]
            # thin lines connecting matched siblings (subsample for legibility)
            n_lines = min(max_lines, k)
            li = np.random.default_rng(0).choice(k, size=n_lines, replace=False)
            for j in li:
                ax.plot([pos_xy[j, 0], neg_xy[j, 0]], [pos_xy[j, 1], neg_xy[j, 1]],
                        color="#999999", lw=0.3, alpha=0.4, zorder=1)
            ax.scatter(pos_xy[:, 0], pos_xy[:, 1], s=7, c=CORRECT_COLOR, alpha=0.6, label="correct sibling", linewidths=0, zorder=2)
            ax.scatter(neg_xy[:, 0], neg_xy[:, 1], s=7, c=ERROR_COLOR, alpha=0.6, label="incorrect sibling", linewidths=0, zorder=2)
            ax.set_title(f"{m['label']} ({method})\n{f1_str(m)}  forks={m['n_forks']}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(method)
    axes[0, 0].legend(loc="upper right", fontsize=8, markerscale=2)
    fig.suptitle("Matched fork siblings (same prefix): correct vs incorrect next step; lines connect siblings")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out_dir / "forks_embeddings.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'forks_embeddings.png'}")

    # Displacement along the probe direction: d = (h_neg - h_pos) . w/|w|.
    # Positive d = the incorrect sibling sits further in the probe's error
    # direction than its own correct sibling (content controlled).
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    for c, m in enumerate(models):
        wn = m["w"] / (np.linalg.norm(m["w"]) + 1e-12)
        d = (m["neg_h"] - m["pos_h"]) @ wn
        frac = float((d > 0).mean())
        ax = axes[0, c]
        ax.hist(d, bins=40, color="#756bb1", alpha=0.8, density=True)
        ax.axvline(0.0, color="k", ls="--", lw=1.4)
        ax.set_title(f"{m['label']}\n{f1_str(m)}\nincorrect>correct: {frac*100:.0f}%", fontsize=9)
        ax.set_xlabel("(neg - pos) . probe direction")
        if c == 0:
            ax.set_ylabel("density")
        _log(f"  {m['label']}: fork displacement>0 in {frac*100:.1f}% of {m['n_forks']} forks")
    fig.suptitle("Per-fork displacement along the probe direction (content-controlled correctness shift)")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out_dir / "forks_displacement.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'forks_displacement.png'}")


def fig_probe(models, out_dir, topk=15):
    """Interpret the LINEAR probe directly on PRM800K val_1k:
      - which hidden units create the correct/incorrect logit gap (is it one unit?),
      - the single most important unit's class-conditional distribution,
      - a decision-aligned 2D view (x = probe direction) showing linear separability.
    """
    n = len(models)
    info = {}

    # A: per-unit contribution c_i = w_i * (mu_incorrect_i - mu_correct_i).
    _log("building probe feature-importance figure ...")
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.2), squeeze=False)
    for c, m in enumerate(models):
        h, y, w = m["h"], m["y"], m["w"]
        mu0 = h[y == 0].mean(0); mu1 = h[y == 1].mean(0)
        contrib = w * (mu1 - mu0)
        tot = float(np.abs(contrib).sum()) + 1e-12
        order = np.argsort(-np.abs(contrib))
        csum = np.cumsum(np.abs(contrib)[order]) / tot
        top_dim = int(order[0]); top_share = float(abs(contrib[top_dim]) / tot)
        n90 = int(np.searchsorted(csum, 0.9)) + 1
        info[c] = (top_dim, top_share, n90)

        ax = axes[0, c]; td = order[:topk]; vals = contrib[td]
        ax.bar(range(len(td)), vals, color=[ERROR_COLOR if v > 0 else CORRECT_COLOR for v in vals])
        ax.set_xticks(range(len(td))); ax.set_xticklabels([str(int(i)) for i in td], rotation=90, fontsize=6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{m['label']}\nunit {top_dim}: {top_share*100:.0f}% | 90% in {n90} units", fontsize=9)
        if c == 0:
            ax.set_ylabel("contribution w_i·Δμ_i\n(+ pushes toward incorrect)")
        ax2 = axes[1, c]
        ax2.plot(np.arange(1, len(csum) + 1), csum, color="#333")
        ax2.axhline(0.9, color="grey", ls=":"); ax2.axvline(n90, color="grey", ls=":")
        ax2.set_xscale("log"); ax2.set_ylim(0, 1)
        ax2.set_xlabel("# hidden units (sorted by |contribution|)")
        if c == 0:
            ax2.set_ylabel("cumulative |contribution|")
    fig.suptitle("Which hidden units create the correct/incorrect logit gap (PRM800K val_1k)")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_dir / "probe_feature_importance.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'probe_feature_importance.png'}")

    # B: most important single unit's distribution by class.
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    for c, m in enumerate(models):
        h, y = m["h"], m["y"]; top_dim, top_share, _ = info[c]
        x = h[:, top_dim]
        bins = np.linspace(float(x.min()), float(x.max()), 41)
        axes[0, c].hist(x[y == 0], bins=bins, color=CORRECT_COLOR, alpha=0.6, density=True, label="correct")
        axes[0, c].hist(x[y == 1], bins=bins, color=ERROR_COLOR, alpha=0.6, density=True, label="incorrect")
        axes[0, c].set_title(f"{m['label']}  unit {top_dim}\nshare={top_share*100:.0f}%", fontsize=9)
        axes[0, c].set_xlabel("activation value")
        if c == 0:
            axes[0, c].set_ylabel("density"); axes[0, c].legend(fontsize=8)
    fig.suptitle("Most important single hidden unit: value distribution by class")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out_dir / "probe_top_unit.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'probe_top_unit.png'}")

    # C: decision-aligned 2D view (x = probe direction).
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.3), squeeze=False)
    for c, m in enumerate(models):
        h, y, w, b, thr = m["h"], m["y"], m["w"], m["b"], m["thr"]
        nrmw = float(np.linalg.norm(w)) + 1e-12; u = w / nrmw
        pdir = h @ u
        hperp = h - pdir[:, None] * u[None, :]
        q = pca_scores(hperp.astype(np.float32), 1)[:, 0]
        z_thr = np.log(thr / (1 - thr)); p_b = (z_thr - b) / nrmw
        ax = axes[0, c]
        for cls, color, name in [(0, CORRECT_COLOR, "correct"), (1, ERROR_COLOR, "incorrect")]:
            msk = y == cls
            ax.scatter(pdir[msk], q[msk], s=7, c=color, alpha=0.5, linewidths=0, label=name)
        ax.axvline(p_b, color="k", ls="--", lw=1.4)
        ax.set_title(f"{m['label']}\nval F1={m['val_step_f1']:.3f}", fontsize=9)
        ax.set_xlabel("probe direction (decision axis)"); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("top orthogonal PCA dir"); ax.legend(fontsize=8)
    fig.suptitle("Decision-aligned view: x = probe direction (linear separability); dashed = deployed threshold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out_dir / "decision_axis.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'decision_axis.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, default=Path("runs/s1_model_size_dense"))
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    p.add_argument("--source", choices=["val", "pb", "forks", "probe"], default="val")
    p.add_argument("--subsets", nargs="+", default=SUBSETS)
    p.add_argument("--pb_embed_subset", default="olympiadbench")
    p.add_argument("--forks_stem", default="forks_val_items",
                   help="Stem of the encoded fork items under <tag>/forks/.")
    p.add_argument("--max_forks", type=int, default=800, help="Subsample forks for plotting.")
    p.add_argument("--max_lines", type=int, default=300, help="Sibling connector lines to draw.")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--tsne", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir or (args.runs_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for tag in args.tags:
        md = args.runs_root / tag
        _log(f"loading {tag} ({args.source}) ...")
        if args.source in ("val", "probe"):
            m = load_val(md)
        elif args.source == "pb":
            m = load_pb(md, args.subsets)
        else:
            m = load_forks(md, args.forks_stem, args.max_forks)
        if m is None:
            print(f"[viz] skip {tag}: missing {args.source} data / probe")
            continue
        models.append(m)
        _log(f"{m['label']}: {f1_str(m)}" + (f"  forks={m.get('n_forks')}" if args.source == "forks" else ""))
    if not models:
        raise SystemExit(f"[viz] no {args.source} data under {args.runs_root}")

    if args.source == "val":
        fig_val(models, out_dir, args.tsne)
    elif args.source == "pb":
        fig_pb(models, args.subsets, out_dir, args.pb_embed_subset, args.tsne)
    elif args.source == "probe":
        fig_probe(models, out_dir)
    else:
        fig_forks(models, out_dir, args.tsne, args.max_lines)
    _log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
