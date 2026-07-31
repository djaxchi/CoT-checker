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


def fit_lr(X, y, l2=1.0, iters=200, lr=0.7):
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
    """Per trace: (trace_label, [step_dict, ...]).

    step_dict has, from the identical full-solution states, several representations
    that differ only in how future steps enter (window W steps, -1 = to end):
      cur   = mean(step k tokens)                           (current only)
      pc    = concat[past boundary, cur]                    (+ clean past)
      pcf   = concat[past, cur, mean future STATE]          (future state pool)
      pcd   = concat[past, cur, mean future within-step DELTA]   (future transition)
      pcd2  = concat[past, cur, mean consecutive step-end DELTA] (onset transition)
    The two delta variants strip the persistent "error region" level that made pcf
    collapse localization, keeping only what CHANGED across future steps.
    """
    meta = store.meta()
    out = []
    for k in range(len(store)):
        h = store.item(k)  # (n_tokens, d)
        m = meta[k]
        ss, se, lbl, ns = m["step_starts"], m["step_ends"], m["label"], m["n_steps"]
        end = [h[se[j] - 1] for j in range(ns)]                       # last-token state
        preb = [h[ss[j] - 1] if ss[j] - 1 >= 0 else np.zeros(h.shape[1], np.float32)
                for j in range(ns)]
        cur_pool = [h[ss[j]:se[j]].mean(0) for j in range(ns)]
        within = [end[j] - preb[j] for j in range(ns)]               # within-step delta
        steps = []
        for j in range(ns):
            cur = cur_pool[j]; past = preb[j]
            hi = (ns - 1) if W == -1 else min(j + W, ns - 1)
            futs = list(range(j + 1, hi + 1))
            z = np.zeros_like(cur)
            f_state = np.mean([cur_pool[f] for f in futs], 0) if futs else z
            f_within = np.mean([within[f] for f in futs], 0) if futs else z
            f_consec = np.mean([end[f] - end[f - 1] for f in futs], 0) if futs else z
            steps.append({
                "cur": cur,
                "pc": np.concatenate([past, cur]),
                "pcf": np.concatenate([past, cur, f_state]),
                "pcd": np.concatenate([past, cur, f_within]),
                "pcd2": np.concatenate([past, cur, f_consec]),
                "label": int(j == lbl),
            })
        out.append((lbl, steps))
    return out


def cv_eval(traces, key, k_folds=5, seed=0):
    """Out-of-fold AUROC + mean F1_PB for one representation key (see build_...).

    Vectorized: flatten all steps once, score a whole split with a single matmul
    per fold (no per-step Python matmuls)."""
    # flatten steps in trace/step order; remember each trace's [start,end) + label
    feats, labs, tr_of, spans = [], [], [], []
    for ti, (lbl, steps) in enumerate(traces):
        s0 = len(feats)
        for st in steps:
            feats.append(st[key]); labs.append(st["label"]); tr_of.append(ti)
        spans.append((s0, len(feats), lbl))
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labs, dtype=np.float32)
    tr_of = np.asarray(tr_of)

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(traces)), k_folds)
    grid = np.round(np.arange(0.10, 0.95, 0.05), 2)
    oof = np.zeros(len(y)); f1s = []
    for f in range(k_folds):
        te = set(folds[f].tolist())
        test_mask = np.array([t in te for t in tr_of])
        mu = X[~test_mask].mean(0); sd = X[~test_mask].std(0) + 1e-6
        Xs = (X - mu) / sd
        w, b = fit_lr(Xs[~test_mask], y[~test_mask])
        S = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        oof[test_mask] = S[test_mask]
        tr_traces = [(lbl, S[s0:e0]) for ti, (s0, e0, lbl) in enumerate(spans) if ti not in te]
        te_traces = [(lbl, S[s0:e0]) for ti, (s0, e0, lbl) in enumerate(spans) if ti in te]
        t_star = max(grid, key=lambda t: f1_pb(tr_traces, float(t)))
        f1s.append(f1_pb(te_traces, float(t_star)))
    return auroc(y, oof), float(np.mean(f1s))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store_root", required=True, type=Path,
                   help="full-solution store root; subsets are subdirs")
    p.add_argument("--subsets", nargs="+", default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--windows", nargs="+", type=int, default=[1, 2, -1],
                   help="future horizons W (steps); -1 = to end")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    hdr = (f"{'subset':14s} {'W':>3s}  {'cur':>6s} {'pc':>6s} {'pcf':>6s} {'pcd':>6s} "
           f"{'pcd2':>6s}   {'curF1':>6s} {'pcdF1':>6s} {'pcd2F1':>6s}")
    print("[AUROC for cur/pc/pcf/pcd/pcd2, then F1_PB for cur/pcd/pcd2]", flush=True)
    print(hdr, flush=True)
    for sub in args.subsets:
        store = ShardedRepSplit(args.store_root / sub)
        # cur and pc are future-independent: compute once per subset.
        base = build_trace_examples(store, W=1)
        if args.limit:
            base = base[:args.limit]
        cur_auc, cur_f1 = cv_eval(base, "cur")
        pc_auc, _ = cv_eval(base, "pc")
        for W in args.windows:
            traces = base if W == 1 else build_trace_examples(store, W)
            if args.limit:
                traces = traces[:args.limit]
            pcf_auc, _ = cv_eval(traces, "pcf")
            pcd_auc, pcd_f1 = cv_eval(traces, "pcd")
            pcd2_auc, pcd2_f1 = cv_eval(traces, "pcd2")
            tag = "all" if W == -1 else str(W)
            print(f"{sub:14s} {tag:>3s}  {cur_auc:6.3f} {pc_auc:6.3f} {pcf_auc:6.3f} "
                  f"{pcd_auc:6.3f} {pcd2_auc:6.3f}   {cur_f1:6.3f} {pcd_f1:6.3f} {pcd2_f1:6.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
