"""Gate G0 of step_bert_probe_v0: can a step be linearly read off its neighbours?

Layer-35 states are causal, so every state after step i has attended to step i. If
step i's pooled vector can be reconstructed almost perfectly from the NEXT step's
vectors, masked-step pretraining with right context is a copy task with no
correctness signal. This measures held-out ridge R^2 for predicting step i from one
neighbour, on two token definitions:

    state         target m_i (span mean);       neighbours b_{i+1}, m_{i+1} | b_i, m_{i-1}
    contribution  target c_i = m_i - b_i;        neighbours c_{i+1} | c_{i-1}

Inputs and targets are reduced by PCA (fit on the train problems) before ridge.
R^2 is variance-weighted over target components. Rule from the plan: right-side
R^2 > 0.9 means the raw tokens leak.

It also prints a linear PREVIEW of G1: AUROC of each step's held-out residual norm
for incorrect vs correct. This is a single linear neighbour, not the pretrained
encoder, so it is only a hint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

ALPHAS = np.logspace(-2, 4, 13)


def neighbour_pairs(trace_id: np.ndarray, step_idx: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """(target row i, neighbour row j) with step_j = step_i + offset in the same trace."""
    pos = {(t, int(s)): r for r, (t, s) in enumerate(zip(trace_id.tolist(), step_idx.tolist()))}
    tgt, nb = [], []
    for r, (t, s) in enumerate(zip(trace_id.tolist(), step_idx.tolist())):
        j = pos.get((t, int(s) + offset))
        if j is not None:
            tgt.append(r)
            nb.append(j)
    return np.array(tgt, dtype=np.int64), np.array(nb, dtype=np.int64)


def is_test_group(groups: np.ndarray, frac: float = 0.2) -> np.ndarray:
    h = np.array([int(hashlib.sha256(str(g).encode()).hexdigest()[:8], 16) for g in groups.tolist()])
    return (h % 1000) < int(frac * 1000)


def _fit(x, y, groups, n_comp=None):
    te = is_test_group(groups)
    tr = ~te
    if n_comp is not None:
        px = PCA(n_comp, random_state=0).fit(x[tr])
        py = PCA(n_comp, random_state=0).fit(y[tr])
        x, y = px.transform(x), py.transform(y)
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    x = (x - mu) / sd
    model = RidgeCV(alphas=ALPHAS).fit(x[tr], y[tr])
    pred = model.predict(x[te])
    ym = y[tr].mean(0)
    r2 = 1.0 - ((y[te] - pred) ** 2).sum() / ((y[te] - ym) ** 2).sum()
    resid = np.linalg.norm(y[te] - pred, axis=1)
    return float(r2), te, resid, float(model.alpha_)


def ridge_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_comp: int | None = None) -> float:
    return _fit(x, y, groups, n_comp)[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pooled", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_comp", type=int, default=256)
    args = p.parse_args()

    z = np.load(args.pooled)
    b, m = z["b"].astype(np.float32), z["m"].astype(np.float32)
    c = m - b
    tid, step, y, grp = z["trace_id"], z["step_idx"], z["y"], z["problem_id"]
    nxt = neighbour_pairs(tid, step, +1)
    prv = neighbour_pairs(tid, step, -1)

    # (token def, side, neighbour name, target array, neighbour array, pairs)
    cells = [
        ("state", "right", "b_{i+1}", m, b, nxt),
        ("state", "right", "m_{i+1}", m, m, nxt),
        ("state", "left", "b_i (own boundary)", m, b, (np.arange(len(m)), np.arange(len(m)))),
        ("state", "left", "m_{i-1}", m, m, prv),
        ("contribution", "right", "c_{i+1}", c, c, nxt),
        ("contribution", "left", "c_{i-1}", c, c, prv),
    ]
    rows = []
    for tok, side, name, tgt_arr, nb_arr, (ti, nj) in cells:
        r2, te, resid, alpha = _fit(nb_arr[nj], tgt_arr[ti], grp[ti], args.n_comp)
        yt = y[ti][te]
        auc = float(roc_auc_score(yt, resid)) if 0 < yt.mean() < 1 else float("nan")
        rows.append({"tokens": tok, "side": side, "neighbour": name, "n_pairs": int(len(ti)),
                     "n_test": int(te.sum()), "r2": round(r2, 4), "alpha": alpha,
                     "preview_auroc_resid": round(auc, 4)})
        print(f"{tok:13s} {side:5s} {name:20s} n={len(ti):6d} R2={r2:.3f} "
              f"alpha={alpha:g} residAUROC={auc:.3f}", flush=True)

    out = {"pooled": str(args.pooled), "n_steps": int(len(y)),
           "n_traces": int(len(set(tid.tolist()))), "n_comp": args.n_comp,
           "leak_threshold": 0.9, "cells": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
