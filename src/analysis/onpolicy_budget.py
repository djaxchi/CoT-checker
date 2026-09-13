"""Sampling under a budget: when to stop, and what to answer.

Self-consistency has a blind spot that is easy to miss when every method is
evaluated at a fixed N=10. With one sample there is no vote at all, with two
there is either agreement or a coin flip, and nothing a vote-based rule can
compute distinguishes a good first sample from a bad one. A verifier can: it
scores a single trajectory. So the place to look for a use is not N=10, where
counting wins (REPORT.md §20.2), but the low-budget regime where counting has
nothing to count.

Two decisions are kept separate here on purpose, because conflating them is how
the earlier best-of-N result hid what it was doing:

  the stopping rule   how many samples to draw for this problem
  the answer rule     which answer to return given the samples drawn

A verifier can help with either, and the 2x2 says which. `simulate` replays a
problem's saved candidates in a random order and records the state after each
draw, so any (stopping, answer) pair can be scored offline against the same
draws.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

# Features available to a stopping rule after t draws. Everything here is
# computable at decision time; nothing reads a candidate that has not been drawn.
VOTE_FEATURES = ["t", "top_frac", "margin_frac", "distinct_frac", "entropy"]
SCORE_FEATURES = ["best_q", "mean_q", "lead_best_q", "lead_mean_q", "lead_minus_all"]

ANSWER_MAJORITY = "majority"          # ties resolved by chance, scored as expectation
ANSWER_VERIFIER_TIE = "majority+tie"  # ties resolved by the verifier
ANSWER_CHEAP_TIE = "majority+cheap"   # ties resolved by a free ordering (length)


def draw_state(answers: Sequence, correct: Sequence, quality: Sequence,
               order: Sequence[int], t: int,
               cheap: Sequence | None = None) -> dict | None:
    """The decision state after the first `t` draws of `order`.

    `quality` is higher-is-better, so a verifier's suspicion enters negated. An
    answer that did not parse is drawn and paid for but cannot be voted for.
    `cheap` is a free ordering of the same candidates, the control that says
    whether the verifier is doing anything a ruler could not.
    """
    seq = list(order[:t])
    groups: dict[object, list[int]] = defaultdict(list)
    for i in seq:
        if answers[i] is not None:
            groups[answers[i]].append(i)
    if not groups:
        return None
    items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    top = len(items[0][1])
    second = len(items[1][1]) if len(items) > 1 else 0
    tied = [a for a, v in items if len(v) == top]
    # The two answer rules, scored on the same draws.
    by_chance = float(np.mean([bool(correct[groups[a][0]]) for a in tied]))
    by_verifier = max(tied, key=lambda a: max(quality[i] for i in groups[a]))
    by_cheap = (max(tied, key=lambda a: max(cheap[i] for i in groups[a]))
                if cheap is not None else by_verifier)
    lead = groups[by_verifier] if len(tied) > 1 else items[0][1]
    q_all = np.array([quality[i] for i in seq], dtype=float)
    q_lead = np.array([quality[i] for i in lead], dtype=float)
    share = np.array([len(v) for _, v in items], dtype=float)
    share /= share.sum()
    return {
        "t": t,
        ANSWER_MAJORITY: by_chance,
        ANSWER_VERIFIER_TIE: float(bool(correct[groups[by_verifier][0]])),
        ANSWER_CHEAP_TIE: float(bool(correct[groups[by_cheap][0]])),
        # Whether the verifier is consulted at all, and how many candidates it
        # would have to score if it is only run when the vote cannot decide.
        "tied": len(tied) > 1,
        "n_scored_if_lazy": sum(len(groups[a]) for a in tied) if len(tied) > 1 else 0,
        "vote": [t / 10.0, top / t, (top - second) / t, len(items) / t,
                 float(-(share * np.log(share)).sum())],
        "score": [float(q_all.max()), float(q_all.mean()), float(q_lead.max()),
                  float(q_lead.mean()), float(q_lead.mean() - q_all.mean())],
    }


def simulate(answers: Sequence, correct: Sequence, quality: Sequence,
             members: Sequence[int], n_draws: int, rng: np.random.Generator,
             cheap: Sequence | None = None) -> list[list[dict | None]]:
    """`n_draws` random sampling orders of one problem's candidates."""
    idx = np.asarray(members)
    return [[draw_state(answers, correct, quality, order, t, cheap)
             for t in range(1, len(order) + 1)]
            for order in (rng.permutation(idx) for _ in range(n_draws))]


def features(state: dict, which: str) -> list[float]:
    if which == "vote":
        return state["vote"]
    if which == "score":
        return state["score"]
    return state["vote"] + state["score"]


def apply_threshold(probs: np.ndarray, tau: float) -> int:
    """Index of the first draw whose stopping probability clears tau.

    Never stopping means paying for every candidate, which is the fixed-budget
    behaviour and the right default: a rule that cannot make up its mind has not
    saved anything.
    """
    hit = np.flatnonzero(probs >= tau)
    return int(hit[0]) if hit.size else len(probs) - 1


def frontier_envelope(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """The upper-left envelope: the best accuracy reachable at each cost or less.

    A method is only credited with an operating point it can actually reach, so
    comparing two methods means comparing their envelopes, never their best
    single points.
    """
    out: list[tuple[float, float]] = []
    for cost, acc in sorted(points):
        if not out or acc > out[-1][1]:
            out.append((cost, acc))
    return out


def accuracy_at_budget(envelope: Sequence[tuple[float, float]], budget: float) -> float:
    """Best accuracy the envelope reaches without exceeding `budget` samples."""
    ok = [a for c, a in envelope if c <= budget + 1e-9]
    return max(ok) if ok else float("nan")
