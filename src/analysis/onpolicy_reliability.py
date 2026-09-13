"""Predicting whether the vote is right, rather than which candidate is best.

Best-of-N asks the verifier to rank candidates, and it loses to counting them
(REPORT.md §20.2). That is a statement about ranking, not about the scores. A
different decision is left open: given the votes already cast, is the answer they
point at correct? A caller who can answer that can abstain, escalate to a larger
model, or spend more samples where they help, and none of that requires the
verifier to out-rank the vote.

So this fits P(majority answer is correct) from the vote statistics alone, then
again with the hidden-state scores added, and asks whether the scores bought
anything. The vote-only model is the control that the arm never ran: without it,
any accuracy here would just be rediscovering that agreement predicts
correctness.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

VOTE_FEATURES = ["top_frac", "margin", "n_distinct_frac", "entropy",
                 "ungradeable_frac"]
CHEAP_FEATURES = ["mean_chars", "mean_steps"]
SCORE_FEATURES = ["bloc_mean_score", "bloc_min_score", "bloc_max_score",
                  "bloc_minus_rest"]
FEATURE_SETS = {
    "vote": VOTE_FEATURES,
    "vote+length": VOTE_FEATURES + CHEAP_FEATURES,
    "vote+score": VOTE_FEATURES + SCORE_FEATURES,
    "vote+length+score": VOTE_FEATURES + CHEAP_FEATURES + SCORE_FEATURES,
    "score": SCORE_FEATURES,
}


def vote_features(rows: Sequence[dict]) -> dict[str, float]:
    """Everything derivable from the answers alone, before any model is consulted.

    `margin` is the gap between the top answer's count and the runner-up's, which
    is zero exactly on the co-plurality ties of §20.12.
    """
    n = len(rows)
    counts = Counter(r["answer"] for r in rows if r["answer"] is not None)
    ordered = sorted(counts.values(), reverse=True)
    top = ordered[0] if ordered else 0
    second = ordered[1] if len(ordered) > 1 else 0
    p = np.array(ordered, dtype=float) / max(1, sum(ordered))
    entropy = float(-(p * np.log(p)).sum()) if ordered else 0.0
    return {"top_frac": top / n, "margin": (top - second) / n,
            "n_distinct_frac": len(counts) / n, "entropy": entropy,
            "ungradeable_frac": sum(r["answer"] is None for r in rows) / n}


def score_features(rows: Sequence[dict], bloc: Sequence[int]) -> dict[str, float]:
    """Aggregates of the verifier over the winning bloc, and against the rest.

    `bloc_minus_rest` is the contrast the ranking view throws away: not how
    suspicious the winner is, but how suspicious it is compared with the
    candidates that lost the vote.
    """
    if not bloc:
        return {k: 0.0 for k in SCORE_FEATURES}
    worst = np.array([max(rows[i]["scores"]) for i in bloc], dtype=float)
    rest = [max(rows[i]["scores"]) for i in range(len(rows)) if i not in set(bloc)]
    return {"bloc_mean_score": float(worst.mean()),
            "bloc_min_score": float(worst.min()),
            "bloc_max_score": float(worst.max()),
            "bloc_minus_rest": float(worst.mean() - np.mean(rest)) if rest else 0.0}


def length_features(rows: Sequence[dict], bloc: Sequence[int]) -> dict[str, float]:
    if not bloc:
        return {k: 0.0 for k in CHEAP_FEATURES}
    return {"mean_chars": float(np.mean([rows[i]["n_chars"] for i in bloc])),
            "mean_steps": float(np.mean([rows[i]["n_steps"] for i in bloc]))}


def problem_features(rows: Sequence[dict], bloc: Sequence[int]) -> dict[str, float]:
    return {**vote_features(rows), **length_features(rows, bloc),
            **score_features(rows, bloc)}


def grouped_folds(groups: Sequence[str], n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Fold assignment that keeps a repeated question whole.

    The evaluation pool's 300 ids are 284 question texts. Splitting on ids would
    fit and test on the same question and report a number that is too good.
    """
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[j]: int(k % n_folds) for k, j in enumerate(order)}
    assign = np.array([fold_of[g] for g in groups])
    return [np.flatnonzero(assign == k) for k in range(n_folds)]


def f1_at(y: np.ndarray, p: np.ndarray, t: float) -> float:
    pred = p >= t
    tp = float((pred & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / pred.sum(), tp / (y == 1).sum()
    return 2 * prec * rec / (prec + rec)


def trivial_f1(y: np.ndarray) -> float:
    """Always predicting "the vote is right", the baseline any model must beat."""
    return f1_at(y, np.ones_like(y, dtype=float), 0.5)
