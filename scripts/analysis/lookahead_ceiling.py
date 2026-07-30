#!/usr/bin/env python3
"""Go/no-go ceiling test: does FUTURE-step context improve first-error detection?

Reads a full-solution ProcessBench store (encode_processbench_full_store.py) and,
for every step k, builds pooled representations from the identical last-layer
states, differing only in how much context is included:

    current           = mean(step k tokens)
    past+current      = concat[ state at pre-step boundary, mean(step k tokens) ]
    past+cur+future_W = concat[ past, current, mean(steps k+1..k+W tokens) ]

Then a probe is trained with group cross-validation BY TRACE (no trace spans a
fold boundary) and evaluated on held-out steps. This is a CEILING test, not the
deployable protocol: it trains within ProcessBench purely to answer whether the
future context that ProcessBench makes available carries first-error signal beyond
past+current. Primary metric: out-of-fold per-step AUROC. Secondary: F1_PB with
the first-error threshold picked on each fold's training traces.

numpy only (runs in the offline cluster venv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.repstore.store import ShardedRepSplit  # noqa: E402


def auroc(y, s):
    """Tie-corrected Mann-Whitney AUROC (rank based)."""
    y = np.asarray(y); s = np.asarray(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, counts = np.unique(s, return_counts=True, return_inverse=True)
    csum = np.cumsum(counts)
    avg_rank = csum - (counts - 1) / 2.0          # mean rank of each tie group
    ranks = avg_rank[inv]
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fit_lr(X, y, l2=1.0, iters=500, lr=0.5):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


def f1_pb(traces, t):
    ne = nc = eh = ch = 0
    for label, scores in traces:
        pred = -1
        for i, s in enumerate(scores):
            if s > t:
                pred = i; break
        if label == -1:
            nc += 1; ch += int(pred == -1)
        else:
            ne += 1; eh += int(pred == label)
    ae = eh / ne if ne else 0.0
    ac = ch / nc if nc else 0.0
    return (2 * ae * ac / (ae + ac)) if (ae + ac) else 0.0


def build_trace_examples(store, W):
    """Per trace: (trace_label, [(feat_current, feat_pc, feat_pcf, step_label), ...])."""
    meta = store.meta()
    out = []
    for k in range(len(store)):
        h = store.item(k)  # (n_tokens, d)
        m = meta[k]
        ss, se, lbl, ns = m["step_starts"], m["step_ends"], m["label"], m["n_steps"]
        steps = []
        for j in range(ns):
            cur = h[ss[j]:se[j]].mean(0)
            past = h[ss[j] - 1] if ss[j] - 1 >= 0 else np.zeros_like(cur)
            fe = se[min(j + W, ns - 1)] if W != -1 else se[-1]
            if j + 1 <= ns - 1:
                fut = h[ss[j + 1]:fe].mean(0)
            else:
                fut = np.zeros_like(cur)
            steps.append((cur, np.concatenate([past, cur]),
                          np.concatenate([past, cur, fut]), int(j == lbl)))
        out.append((lbl, steps))
    return out


def cv_eval(traces, rep_key, k_folds=5, seed=0):
    """Out-of-fold AUROC + mean F1_PB for one representation key (0=cur,1=pc,2=pcf)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(traces))
    folds = np.array_split(idx, k_folds)
    oof_y, oof_s = [], []
    f1s = []
    for f in range(k_folds):
        te = set(folds[f].tolist())
        tr_i = [i for i in range(len(traces)) if i not in te]
        te_i = list(folds[f])
        Xtr = np.array([st[rep_key] for i in tr_i for st in traces[i][1]], dtype=np.float32)
        ytr = np.array([st[3] for i in tr_i for st in traces[i][1]], dtype=np.float32)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd
        w, b = fit_lr(Xtr, ytr)
        # threshold on train traces (grid max F1_PB)
        tr_traces = []
        for i in tr_i:
            sc = [1 / (1 + np.exp(-(((st[rep_key] - mu) / sd) @ w + b))) for st in traces[i][1]]
            tr_traces.append((traces[i][0], sc))
        grid = np.round(np.arange(0.05, 1.0, 0.02), 2)
        t_star = max(grid, key=lambda t: f1_pb(tr_traces, float(t)))
        # eval test traces
        te_traces = []
        for i in te_i:
            sc = [float(1 / (1 + np.exp(-(((st[rep_key] - mu) / sd) @ w + b)))) for st in traces[i][1]]
            te_traces.append((traces[i][0], sc))
            for st, s in zip(traces[i][1], sc):
                oof_y.append(st[3]); oof_s.append(s)
        f1s.append(f1_pb(te_traces, float(t_star)))
    return auroc(oof_y, oof_s), float(np.mean(f1s))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store_root", required=True, type=Path,
                   help="full-solution store root; subsets are subdirs")
    p.add_argument("--subsets", nargs="+", default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--windows", nargs="+", type=int, default=[1, 2, -1],
                   help="future horizons W (steps); -1 = to end")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    print(f"{'subset':14s} {'W':>3s}  {'cur_AUC':>8s} {'pc_AUC':>8s} {'pcf_AUC':>8s}  "
          f"{'cur_F1':>7s} {'pc_F1':>7s} {'pcf_F1':>7s}", flush=True)
    for sub in args.subsets:
        store = ShardedRepSplit(args.store_root / sub)
        for W in args.windows:
            traces = build_trace_examples(store, W)
            if args.limit:
                traces = traces[:args.limit]
            cur_auc, cur_f1 = cv_eval(traces, 0)
            pc_auc, pc_f1 = cv_eval(traces, 1)
            pcf_auc, pcf_f1 = cv_eval(traces, 2)
            tag = "all" if W == -1 else str(W)
            print(f"{sub:14s} {tag:>3s}  {cur_auc:8.3f} {pc_auc:8.3f} {pcf_auc:8.3f}  "
                  f"{cur_f1:7.3f} {pc_f1:7.3f} {pcf_f1:7.3f}", flush=True)


if __name__ == "__main__":
    main()
