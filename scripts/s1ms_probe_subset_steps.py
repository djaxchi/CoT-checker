#!/usr/bin/env python3
"""Two analyses on the cached DenseLinear activations (PRM800K), per model size:

PART A - minimal predictive subset
  Rank hidden units by |probe weight|, refit a logistic probe on the top-k units
  (trained on probe_train_40k), and measure macro step F1 on val_1k as a function
  of k. Finds the smallest subset that retains the probe's accuracy and saves the
  chosen unit indices.

PART B - is the error subset shared across reasoning steps?
  Bin examples by step_idx. Per bin compute the correctness direction
  (standardized mean difference incorrect-correct) and its top units, then compare
  bins three ways:
    - direction cosine          (is the same direction used at every step?)
    - top-unit Jaccard overlap  (are the same units important at every step?)
    - cross-step transfer F1    (does a probe trained on step-i errors detect
                                 step-j errors?)
  High off-diagonal on all three => a shared, position-invariant error code.

CPU only; reuses merged/{probe_train_40k,val_1k}_{h.npy,y.npy,meta.jsonl}.
Figures -> <runs_root>/figures/.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DEFAULT_TAGS = ["qwen2_5_1_5b", "qwen2_5_3b", "qwen2_5_7b", "qwen2_5_14b", "qwen2_5_32b"]
DEFAULT_KS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def _log(msg: str) -> None:
    print(f"[subset {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_probe_w(pt_path: Path) -> np.ndarray:
    import torch
    sd = torch.load(pt_path, map_location="cpu", weights_only=False)
    return sd["fc.weight"].detach().cpu().numpy().reshape(-1).astype(np.float64)


def read_meta_field(path: Path, field: str) -> np.ndarray:
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            vals.append(int(json.loads(line)[field]))
    return np.asarray(vals, dtype=int)


def fit_clf(xtr, ytr, seed=42):
    """Standardize (fit on train) then logistic regression. Scaling makes lbfgs
    converge and the cross-step transfer comparable across bins."""
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(sc.transform(xtr), ytr)
    return clf, sc


def eval_f1(clf, sc, xte, yte) -> float:
    return float(f1_score(yte, clf.predict(sc.transform(xte)), average="macro"))


def macro_f1_fit(xtr, ytr, xte, yte, seed=42) -> float:
    clf, sc = fit_clf(xtr, ytr, seed)
    return eval_f1(clf, sc, xte, yte)


def make_bins(step_idx: np.ndarray, y: np.ndarray, min_per_class: int) -> list[tuple[str, np.ndarray]]:
    """Bin by step_idx: 0,1,2,3,4 individually, then 5+; keep bins with enough of each class."""
    bins: list[tuple[str, np.ndarray]] = []
    specs = [("s0", lambda s: s == 0), ("s1", lambda s: s == 1), ("s2", lambda s: s == 2),
             ("s3", lambda s: s == 3), ("s4", lambda s: s == 4), ("s5+", lambda s: s >= 5)]
    for name, fn in specs:
        m = fn(step_idx)
        if int(((y == 0) & m).sum()) >= min_per_class and int(((y == 1) & m).sum()) >= min_per_class:
            bins.append((name, m))
    return bins


def _heat(ax, M, labels, title, vmin, vmax, cmap):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if (M[i, j] - vmin) / (vmax - vmin + 1e-9) > 0.5 else "black")
    return im


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, default=Path("runs/s1_model_size_dense"))
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--subset_k", type=int, default=2048, help="Unit subset size saved to JSON.")
    p.add_argument("--topk_overlap", type=int, default=256, help="Top units per bin for Jaccard.")
    p.add_argument("--transfer_dims", type=int, default=1024, help="Top-|w| units used for transfer fits (speed).")
    p.add_argument("--min_per_class", type=int, default=150)
    p.add_argument("--max_per_bin", type=int, default=3000, help="Cap per-bin train size for transfer.")
    args = p.parse_args()

    out_dir = args.out_dir or (args.runs_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = {}      # tag -> (ks, f1s)
    bin_results = {}  # tag -> dict(labels, cos, jac, transfer)
    subset_json = {}

    for tag in args.tags:
        md = args.runs_root / tag
        need = [md / "merged" / f"{s}" for s in
                ("probe_train_40k_h.npy", "probe_train_40k_y.npy", "probe_train_40k_meta.jsonl",
                 "val_1k_h.npy", "val_1k_y.npy")]
        if not all(f.exists() for f in need) or not (md / "linear_probe.pt").exists():
            _log(f"skip {tag}: missing merged activations/meta or probe")
            continue
        _log(f"loading {tag} ...")
        tr_h = np.load(md / "merged" / "probe_train_40k_h.npy").astype(np.float32)
        tr_y = np.load(md / "merged" / "probe_train_40k_y.npy").astype(int)
        tr_step = read_meta_field(md / "merged" / "probe_train_40k_meta.jsonl", "step_idx")
        va_h = np.load(md / "merged" / "val_1k_h.npy").astype(np.float32)
        va_y = np.load(md / "merged" / "val_1k_y.npy").astype(int)
        w = load_probe_w(md / "linear_probe.pt")
        rank = np.argsort(-np.abs(w))  # units by probe-weight magnitude

        # ---- PART A: F1 vs k ----
        ks, f1s = [], []
        for k in args.ks:
            k = min(k, tr_h.shape[1])
            cols = rank[:k]
            f1 = macro_f1_fit(tr_h[:, cols], tr_y, va_h[:, cols], va_y)
            ks.append(k); f1s.append(f1)
            _log(f"  {tag} k={k:>5}: val macro F1={f1:.4f}")
        sweep[tag] = (ks, f1s)
        subset_json[tag] = {"subset_k": int(min(args.subset_k, tr_h.shape[1])),
                            "unit_indices": [int(i) for i in rank[:min(args.subset_k, tr_h.shape[1])]]}

        # ---- PART B: step sharing (on probe_train_40k) ----
        sigma = tr_h.std(axis=0) + 1e-6
        bins = make_bins(tr_step, tr_y, args.min_per_class)
        if len(bins) < 2:
            _log(f"  {tag}: not enough populated step bins; skipping step-sharing")
        else:
            labels = [b[0] for b in bins]
            dirs = []
            topsets = []
            for _, m in bins:
                mu0 = tr_h[m & (tr_y == 0)].mean(0); mu1 = tr_h[m & (tr_y == 1)].mean(0)
                dstd = (mu1 - mu0) / sigma
                dirs.append(dstd)
                topsets.append(set(np.argsort(-np.abs(dstd))[:args.topk_overlap].tolist()))
            nb = len(bins)
            cos = np.eye(nb); jac = np.eye(nb)
            for i in range(nb):
                for j in range(nb):
                    a, b = dirs[i], dirs[j]
                    cos[i, j] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    inter = len(topsets[i] & topsets[j]); uni = len(topsets[i] | topsets[j])
                    jac[i, j] = inter / uni if uni else 0.0
            # transfer F1 (top-|w| dims, per-bin train/test split)
            cols = rank[:min(args.transfer_dims, tr_h.shape[1])]
            rng = np.random.default_rng(42)
            tr_idx, te_idx = [], []
            for _, m in bins:
                idx = np.where(m)[0]; rng.shuffle(idx)
                cut = int(0.7 * len(idx))
                tr_idx.append(idx[:cut][: args.max_per_bin]); te_idx.append(idx[cut:][: args.max_per_bin])
            transfer = np.zeros((nb, nb))
            for i in range(nb):
                clf, sc = fit_clf(tr_h[tr_idx[i]][:, cols], tr_y[tr_idx[i]])
                for j in range(nb):
                    transfer[i, j] = eval_f1(clf, sc, tr_h[te_idx[j]][:, cols], tr_y[te_idx[j]])
            bin_results[tag] = {"labels": labels, "cos": cos, "jac": jac, "transfer": transfer}
            off = ~np.eye(nb, dtype=bool)
            _log(f"  {tag} step-sharing: mean off-diag cos={cos[off].mean():.3f} "
                 f"jaccard={jac[off].mean():.3f} transferF1={transfer[off].mean():.3f} "
                 f"(diag transferF1={np.diag(transfer).mean():.3f})")

    if not sweep:
        raise SystemExit(f"[subset] no usable models under {args.runs_root}")

    (out_dir / "subset_units.json").write_text(json.dumps(subset_json, indent=2))

    # Figure 1: F1 vs k
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for tag, (ks, f1s) in sweep.items():
        ax.plot(ks, f1s, "o-", label=tag.replace("qwen2_5_", "").replace("_", "."))
    ax.set_xscale("log"); ax.set_xlabel("# hidden units in probe (top by |weight|)")
    ax.set_ylabel("val macro step F1"); ax.set_title("Minimal predictive subset: F1 vs #units")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "subset_f1_vs_k.png", dpi=150); plt.close(fig)
    _log(f"wrote {out_dir/'subset_f1_vs_k.png'}")

    # Figures 2-4: per-model step-sharing heatmaps
    if bin_results:
        for key, title, vlim, cmap, fname in [
            ("cos", "Correctness-direction cosine across steps", (0.0, 1.0), "viridis", "step_sharing_cosine.png"),
            ("jac", f"Top-{args.topk_overlap}-unit Jaccard across steps", (0.0, 1.0), "magma", "step_sharing_overlap.png"),
            ("transfer", "Cross-step transfer macro F1", (0.4, 0.8), "cividis", "step_transfer_f1.png"),
        ]:
            n = len(bin_results)
            fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.2), squeeze=False)
            last_im = None
            for c, (tag, r) in enumerate(bin_results.items()):
                last_im = _heat(axes[0, c], r[key], r["labels"],
                                f"{tag.replace('qwen2_5_','').replace('_','.')}", vlim[0], vlim[1], cmap)
            fig.suptitle(title)
            fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8)
            fig.savefig(out_dir / fname, dpi=150); plt.close(fig)
            _log(f"wrote {out_dir/fname}")

    _log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
