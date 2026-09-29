"""Exact finite-pool selection expectations for downstream diagnostics."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

AGGREGATIONS = ("worst", "mean", "q90", "last", "worst_skip_first", "geomean")
RULES = ("rerank", "tiebreak", "wvote", "safe_wvote", "group_mean", "gate_margin02")


def stable_fold(question_hash: str, folds: int = 5) -> int:
    """Keep the same question in the same fold across temperatures and traces."""
    return int(hashlib.sha256(("downstream_audit_v1:" + question_hash).encode()).hexdigest()[:16], 16) % folds


def aggregate(scores: list[float], method: str) -> float:
    """Aggregate suspicion, retaining the convention that lower is better."""
    s = np.asarray(scores, float)
    if s.size == 0 or not np.isfinite(s).all():
        raise ValueError("Step scores must be nonempty and finite")
    if np.any((s < 0) | (s > 1)):
        raise ValueError("Expected suspicion probabilities in [0, 1]")
    if method == "worst":
        return float(s.max())
    if method == "mean":
        return float(s.mean())
    if method == "q90":
        return float(np.quantile(s, .9))
    if method == "last":
        return float(s[-1])
    if method == "worst_skip_first":
        return float(s[1:].max() if len(s) > 1 else s[0])
    if method == "geomean":
        return float(1 - np.exp(np.log(np.clip(1-s, 1e-12, 1)).mean()))
    raise ValueError(method)


def describe_pool(keys: list, correct: list[bool]) -> dict:
    """Describe plurality, ties and strict rescue cases without using score order."""
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        if key is not None:
            groups[key].append(i)
    idx = [np.array(v, dtype=int) for v in groups.values()]
    ok = np.asarray(correct, bool)
    if any(len(set(ok[g])) != 1 for g in idx):
        raise ValueError("Inconsistent correctness within an answer identity")
    if any(key is None and flag for key, flag in zip(keys, correct)):
        raise ValueError("Correct candidate has no answer identity")
    counts = np.array([len(g) for g in idx])
    truth = np.array([ok[g[0]] for g in idx], dtype=float)
    top = np.flatnonzero(counts == counts.max()) if len(idx) else np.array([], int)
    majority = float(truth[top].mean()) if len(top) else 0.
    if ok.all():
        case = "unanimous_correct"
    elif not ok.any():
        case = "no_correct_sample"
    elif majority == 1:
        case = "majority_correct_mixed"
    elif majority > 0:
        case = "vote_tie_with_correct"
    else:
        case = "strict_minority_correct"
    sorted_counts = sorted(counts, reverse=True)
    margin = (sorted_counts[0] - (sorted_counts[1] if len(sorted_counts) > 1 else 0))/len(ok) if len(idx) else 0.
    return {"groups": idx, "counts": counts, "truth": truth, "top": top,
            "correct": ok, "majority": majority, "oracle": float(ok.any()),
            "case": case, "margin": margin}


def outcome(pool: dict, suspicion: np.ndarray, rule: str) -> float:
    """Expected accuracy under uniform exact-score ties; no label-dependent choice."""
    s = np.asarray(suspicion, float)
    if not np.isfinite(s).all():
        raise ValueError("Finite candidate scores required")
    if rule == "rerank":
        return float(pool["correct"][s == s.min()].mean())
    if rule == "gate_margin02":
        return outcome(pool, s, "rerank") if pool["margin"] <= .2 else pool["majority"]
    groups = pool["groups"]
    if not groups:
        return 0.
    if rule == "tiebreak":
        eligible = pool["top"]
        values = np.array([-s[groups[i]].min() for i in eligible])
        return float(pool["truth"][eligible[values == values.max()]].mean())
    if rule == "group_mean":
        values = np.array([-s[g].mean() for g in groups])
    elif rule in ("wvote", "safe_wvote"):
        reward = 1-s if rule == "wvote" else .5 + .5*(1-s)
        values = np.array([reward[g].sum() for g in groups])
        if values.max() <= 0:
            return pool["majority"]
    else:
        raise ValueError(rule)
    return float(pool["truth"][values == values.max()].mean())


def within_auc(correct: np.ndarray, suspicion: np.ndarray) -> float:
    """Problem-macro AUROC with half credit for equal score pairs."""
    pos, neg = suspicion[correct], suspicion[~correct]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] < neg[None, :]) + .5*(pos[:, None] == neg[None, :])).mean())
