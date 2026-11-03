"""Threshold-free first-error metrics on ProcessBench (InStep paper V1, Table 2).

Per subset, the items are: for an erroneous trace (first_error = k >= 0) its steps 0..k, the
step k being the only positive; for a correct trace (first_error = -1) all its steps, all
negative. Steps after the first error are dropped (their label is undefined under ProcessBench's
first-error protocol). Metrics: AUROC and AP (PR-AUC of the first-error step) per subset, and
their mean over the four subsets. Seeds: the metric is computed per seed and averaged.

Paired bootstrap: traces are resampled within each subset (the same resample for both arms),
every seed of both arms is re-scored on the resample, delta = mean-over-subsets(A) - (B).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PB = ("gsm8k", "math", "olympiadbench", "omnimath")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def fast_ap(y: np.ndarray, s: np.ndarray) -> float:
    """Step-wise average precision (as sklearn, ties grouped)."""
    npos = y.sum()
    if npos == 0 or npos == len(y):
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    tp = np.cumsum(y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp, n = tp[last], last + 1
    prec, rec = tp / n, tp / npos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC via average ranks (ties half), equal to sklearn's roc_auc_score."""
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    ranks = np.empty(len(s))
    # average rank within tie groups
    start = np.r_[0, np.nonzero(np.diff(ss))[0] + 1]
    end = np.r_[start[1:], len(ss)]
    avg = (start + end + 1) / 2.0  # 1-based average rank of each group
    ranks[o] = np.repeat(avg, end - start)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load_items(data: Path) -> dict:
    """{subset: {"uids", "y", "trace"}} of the first-error items, from data/pb/<sub>.traces.jsonl."""
    out = {}
    for sub in PB:
        p = data / "pb" / f"{sub}.traces.jsonl"
        if not p.exists():
            continue
        uids, y, tr = [], [], []
        for ti, t in enumerate(jl(p)):
            fe = int(t["first_error"])
            n = len(t["steps"])
            last = n - 1 if fe < 0 else min(fe, n - 1)
            for k in range(last + 1):
                uids.append(f"{t['trace_id']}::{k}")
                y.append(1 if k == fe else 0)
                tr.append(ti)
        out[sub] = {"uids": uids, "y": np.array(y, dtype=int), "trace": np.array(tr)}
    return out


def seed_arrays(items: dict, seed_scores: list[dict]) -> dict | None:
    """{subset: [array per seed]} aligned to the items; None if any seed misses any item."""
    out = {}
    for sub, it in items.items():
        arrs = []
        for sc in seed_scores:
            if sc is None or sub not in sc:
                return None
            d = sc[sub]
            try:
                arrs.append(np.array([d[u] for u in it["uids"]], dtype=float))
            except KeyError:
                return None
        out[sub] = arrs
    return out


def point(items: dict, S: dict) -> dict:
    """Per-subset and mean AUROC / AP, seed-averaged; per-seed means kept."""
    res = {"per_subset": {}}
    per_seed_auc, per_seed_ap = [], []
    nseed = len(next(iter(S.values())))
    for sub, it in items.items():
        y = it["y"]
        aucs = [fast_auroc(y, x) for x in S[sub]]
        aps = [fast_ap(y, x) for x in S[sub]]
        res["per_subset"][sub] = {"auroc": float(np.mean(aucs)), "ap": float(np.mean(aps)),
                                  "n_items": int(len(y)), "n_pos": int(y.sum()),
                                  "prevalence": float(y.mean())}
        per_seed_auc.append(aucs)
        per_seed_ap.append(aps)
    res["auroc_mean"] = float(np.mean([v["auroc"] for v in res["per_subset"].values()]))
    res["ap_mean"] = float(np.mean([v["ap"] for v in res["per_subset"].values()]))
    res["seeds_auroc_mean"] = [float(np.mean([a[i] for a in per_seed_auc])) for i in range(nseed)]
    res["seeds_ap_mean"] = [float(np.mean([a[i] for a in per_seed_ap])) for i in range(nseed)]
    res["n_seeds"] = nseed
    return res


def boot_indices(items: dict, n_boot: int, seed: int = 0) -> dict:
    """{subset: list of n_boot item-index arrays}, traces resampled with replacement."""
    rng = np.random.default_rng(seed)
    out = {}
    for sub, it in items.items():
        tr = it["trace"]
        order = np.argsort(tr, kind="mergesort")
        uniq, start = np.unique(tr[order], return_index=True)
        end = np.r_[start[1:], len(tr)]
        groups = [order[a:b] for a, b in zip(start, end)]
        out[sub] = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                    for _ in range(n_boot)]
    return out


def _boot_means(items: dict, S: dict, idx: dict, metric) -> np.ndarray:
    """(n_boot,) mean over subsets of the seed-averaged metric on each resample."""
    subs = list(items)
    nb = len(idx[subs[0]])
    out = np.zeros(nb)
    for sub in subs:
        y = items[sub]["y"]
        for b, ii in enumerate(idx[sub]):
            out[b] += np.mean([metric(y[ii], x[ii]) for x in S[sub]]) / len(subs)
    return out


def boot_arrays(items: dict, S: dict, idx: dict) -> dict:
    """{"auroc_mean": (n_boot,), "ap_mean": (n_boot,)} of one arm on the shared resamples."""
    return {"auroc_mean": _boot_means(items, S, idx, fast_auroc), "ap_mean": _boot_means(items, S, idx, fast_ap)}


def ci(d: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def pair_from_arrays(pa: dict, pb: dict, ba: dict, bb: dict) -> dict:
    """A - B from two arms' point() results and boot_arrays() on the same resamples."""
    out = {}
    for name in ("auroc_mean", "ap_mean"):
        d = ba[name] - bb[name]
        out[name] = {"delta": pa[name] - pb[name], "ci95": ci(d), "share_le0": float((d <= 0).mean())}
    return out


def boot_arm(items: dict, S: dict, idx: dict) -> dict:
    """Bootstrap 95% CI of one arm's mean AUROC and mean AP."""
    return {k: ci(v) for k, v in boot_arrays(items, S, idx).items()}


def boot_pair(items: dict, SA: dict, SB: dict, idx: dict) -> dict:
    """Paired trace bootstrap of A - B on mean AUROC and mean AP (same resamples)."""
    return pair_from_arrays(point(items, SA), point(items, SB),
                            boot_arrays(items, SA, idx), boot_arrays(items, SB, idx))
