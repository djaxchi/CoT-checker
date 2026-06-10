#!/usr/bin/env python3
"""Phase 1: where does the correctness signal live? Per-layer probe + subset.

For each layer in a multi-layer cache (encode_prm800k_multilayer.py), train the
same DenseLinear-style probe (standardized logistic) on probe_train_40k and
evaluate macro step F1 on val_1k, then sweep the top-k most important units to
find the minimal sufficient subset at that layer. Produces:

  layer_f1_vs_depth.png      F1 vs layer fraction (settles "do mid layers carry
                             correctness as well as the last layer?")
  layer_subset_f1_vs_k.png   F1 vs #units, one line per layer
  layer_sweep.json           per-layer F1, chosen k*, and the unit indices
                             (these feed the steering experiment)

CPU only. Reuses merged multi-layer cache; needs sklearn + matplotlib.
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

DEFAULT_KS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def _log(m): print(f"[sweep {time.strftime('%H:%M:%S')}] {m}", flush=True)

SUBSETS = ["gsm8k", "math", "olympiadbench", "omnimath"]


def fit_eval(xtr, ytr, xte, yte):
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(xtr), ytr)
    f1 = float(f1_score(yte, clf.predict(sc.transform(xte)), average="macro"))
    return f1, clf


# --- F1_PB helpers (faithful copies of the project's ProcessBench evaluator) ---

def select_threshold_bacc(scores, y, grid):
    """Balanced-accuracy-optimal threshold over the 0.1..1.0 grid (Sprint 1)."""
    best_t, best_b = grid[0], -1.0
    for t in grid:
        pred = scores > t
        tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        bacc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        if bacc > best_b:
            best_b, best_t = bacc, t
    return best_t


def f1_pb(scores, meta, threshold):
    """Official ProcessBench first-error F1_PB at a threshold (P(error) scores)."""
    groups, labels = {}, {}
    for i, r in enumerate(meta):
        groups.setdefault(r["id"], []).append((int(r["step_idx"]), float(scores[i])))
        labels[r["id"]] = int(r["label"])
    n_err = n_cor = ae = ac = 0
    for tid, items in groups.items():
        items.sort(key=lambda x: x[0])
        pred = -1
        for idx, s in items:
            if s > threshold:
                pred = idx; break
        lab = labels[tid]
        if lab == -1:
            n_cor += 1; ac += (pred == -1)
        else:
            n_err += 1; ae += (pred == lab)
    acc_e = ae / max(n_err, 1) if n_err else 0.0
    acc_c = ac / max(n_cor, 1) if n_cor else 0.0
    return (2 * acc_e * acc_c / (acc_e + acc_c)) if (acc_e + acc_c) > 0 else 0.0


def oracle_f1_pb(scores, meta, step=0.005):
    grid = [round(step * i, 6) for i in range(1, int(round(1.0 / step)))]
    best = 0.0
    for t in grid:
        f = f1_pb(scores, meta, t)
        if f > best:
            best = f
    return best


def read_pb_meta(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", type=Path, required=True, help="Multi-layer cache dir.")
    p.add_argument("--train_stem", default="probe_train_40k")
    p.add_argument("--val_stem", default="val_1k")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--tol", type=float, default=0.01, help="k* = smallest k within tol of full-layer F1.")
    p.add_argument("--pb_cache_dir", type=Path, default=None,
                   help="Dir with <subset>/pb_step_L{idx}_h.npy + pb_step_meta.jsonl. "
                        "If set, also compute per-layer ProcessBench F1_PB (val + oracle).")
    p.add_argument("--subsets", nargs="+", default=SUBSETS)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cache_dir / "multilayer_manifest.json").read_text())
    layer_idx_by_frac = manifest["layer_indices"]  # {"0.10": idx, ...}
    fracs = sorted(layer_idx_by_frac.keys(), key=float)

    ytr = np.load(args.cache_dir / f"{args.train_stem}_y.npy").astype(int)
    yte = np.load(args.cache_dir / f"{args.val_stem}_y.npy").astype(int)

    grid01 = [round(0.1 * i, 1) for i in range(1, 11)]
    per_layer = {}
    for fr in fracs:
        li = layer_idx_by_frac[fr]
        xtr = np.load(args.cache_dir / f"{args.train_stem}_L{li}_h.npy").astype(np.float32)
        xte = np.load(args.cache_dir / f"{args.val_stem}_L{li}_h.npy").astype(np.float32)
        sc = StandardScaler().fit(xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(xtr), ytr)
        full_f1 = float(f1_score(yte, clf.predict(sc.transform(xte)), average="macro"))
        rank = np.argsort(-np.abs(clf.coef_.reshape(-1)))
        ks, f1s = [], []
        for k in args.ks:
            k = min(k, xtr.shape[1]); cols = rank[:k]
            f1k, _ = fit_eval(xtr[:, cols], ytr, xte[:, cols], yte)
            ks.append(k); f1s.append(f1k)
        kstar = next((k for k, f in zip(ks, f1s) if f >= full_f1 - args.tol), ks[-1])
        rec = {"layer_index": li, "full_f1": full_f1, "ks": ks, "f1s": f1s,
               "kstar": kstar, "unit_indices_kstar": [int(i) for i in rank[:kstar]]}

        # Per-layer ProcessBench F1_PB (val-selected threshold + per-subset oracle).
        if args.pb_cache_dir is not None:
            val_scores = clf.predict_proba(sc.transform(xte))[:, 1]
            t_val = select_threshold_bacc(val_scores, yte, grid01)
            val_f1s, or_f1s = {}, {}
            for sub in args.subsets:
                hp = args.pb_cache_dir / sub / f"pb_step_L{li}_h.npy"
                mp = args.pb_cache_dir / sub / "pb_step_meta.jsonl"
                if not (hp.exists() and mp.exists()):
                    continue
                pbh = np.load(hp).astype(np.float32)
                meta = read_pb_meta(mp)
                scores = clf.predict_proba(sc.transform(pbh))[:, 1]
                val_f1s[sub] = f1_pb(scores, meta, t_val)
                or_f1s[sub] = oracle_f1_pb(scores, meta)
            if val_f1s:
                rec["t_val"] = t_val
                rec["pb_val_f1_per_subset"] = val_f1s
                rec["pb_oracle_f1_per_subset"] = or_f1s
                rec["pb_macro_val"] = float(np.mean(list(val_f1s.values())))
                rec["pb_macro_oracle"] = float(np.mean(list(or_f1s.values())))

        per_layer[fr] = rec
        extra = (f"  PB macro val={rec.get('pb_macro_val', float('nan')):.4f} "
                 f"oracle={rec.get('pb_macro_oracle', float('nan')):.4f}"
                 if "pb_macro_val" in rec else "")
        _log(f"frac={fr} L{li}: val stepF1={full_f1:.4f}  k*={kstar}{extra}")

    (args.out_dir / "layer_sweep.json").write_text(json.dumps(
        {"model_name": manifest.get("model_name"), "per_layer": per_layer}, indent=2))

    # F1 vs depth
    xs = [float(fr) for fr in fracs]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(xs, [per_layer[fr]["full_f1"] for fr in fracs], "o-", color="#225522")
    ax.set_xlabel("layer depth (fraction)"); ax.set_ylabel("val macro step F1")
    ax.set_title(f"Correctness F1 vs layer depth  ({manifest.get('model_name','')})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "layer_f1_vs_depth.png", dpi=150); plt.close(fig)
    _log(f"wrote {args.out_dir/'layer_f1_vs_depth.png'}")

    # subset F1 vs k, one line per layer
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for i, fr in enumerate(fracs):
        d = per_layer[fr]
        ax.plot(d["ks"], d["f1s"], "o-", color=cmap(i / max(len(fracs) - 1, 1)), label=f"{fr}")
    ax.set_xscale("log"); ax.set_xlabel("# units (top by |weight|)"); ax.set_ylabel("val macro step F1")
    ax.set_title("Minimal subset per layer"); ax.legend(fontsize=7, title="depth"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "layer_subset_f1_vs_k.png", dpi=150); plt.close(fig)
    _log(f"wrote {args.out_dir/'layer_subset_f1_vs_k.png'}")

    # F1_PB vs depth (the benchmark validation of the val-step-F1 curve)
    if any("pb_macro_val" in per_layer[fr] for fr in fracs):
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(xs, [per_layer[fr]["full_f1"] for fr in fracs], "o:", color="#225522",
                alpha=0.6, label="PRM800K val step F1 (ref)")
        ax.plot(xs, [per_layer[fr].get("pb_macro_oracle", np.nan) for fr in fracs], "s-",
                color="#7a0177", label="ProcessBench macro F1_PB (oracle)")
        ax.plot(xs, [per_layer[fr].get("pb_macro_val", np.nan) for fr in fracs], "^--",
                color="#c51b8a", label="ProcessBench macro F1_PB (val-selected)")
        ax.set_xlabel("layer depth (fraction)"); ax.set_ylabel("F1")
        ax.set_title(f"ProcessBench F1_PB vs layer depth  ({manifest.get('model_name','')})")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(args.out_dir / "layer_f1pb_vs_depth.png", dpi=150); plt.close(fig)
        _log(f"wrote {args.out_dir/'layer_f1pb_vs_depth.png'}")
    _log("done")


if __name__ == "__main__":
    main()
