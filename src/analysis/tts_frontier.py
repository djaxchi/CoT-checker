"""Selection rules over a pool of sampled solutions, as functions of budget.

tts_roster_v1 asks one question: given a budget of generated tokens, what is the
best thing to do with it. Every arm therefore has to land on one axis, and the
axis is tokens rather than N, because a rule that consults a 7B PRM and a rule
that reads states the sampler already computed do not cost the same at equal N.

Nothing here aggregates over problems. A rule applied to one problem's drawn
candidates returns one outcome and one cost, and the caller decides how to
summarise. That is what lets the explorer add a rule later without regenerating
anything (docs/tts_roster_v1_plan.md §2).

Two conventions worth stating because they are easy to get backwards:

**Verifier scores are suspicion.** Higher means more likely wrong, so a selector
minimises them. Confidence statistics are the opposite and are negated on the
way in by `quality_from`, so every rule here maximises quality.

**Majority voting is scored as an expectation over its ties, not by drawing.**
A tied vote broken by a coin has a definite expected accuracy, and taking it
directly keeps the baseline free of sampling noise that would otherwise have to
be averaged away.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

# A candidate carries: answer (normalised, or None), correct (bool),
# tokens (int), and quality (float, higher is better) per scorer.


def quality_from(values: Sequence[float], higher_is_better: bool) -> np.ndarray:
    """Put a scorer's raw output on a common higher-is-better scale.

    A nan becomes -inf rather than 0.0: a candidate whose statistic failed to
    compute must lose every comparison, not sit in the middle of the pack.
    """
    v = np.asarray(values, dtype=float)
    out = np.where(np.isfinite(v), v if higher_is_better else -v, -np.inf)
    return out.astype(float)


def vote(answers: Sequence, drawn: Sequence[int]) -> tuple[dict, list, int]:
    """Answer groups, the tied-at-top answers, and the top count.

    Candidates whose answer did not parse are drawn and paid for but cannot be
    voted for, which is the same convention §20.14's frontier used.
    """
    groups: dict[object, list[int]] = defaultdict(list)
    for i in drawn:
        if answers[i] is not None:
            groups[answers[i]].append(i)
    if not groups:
        return {}, [], 0
    top = max(len(v) for v in groups.values())
    return dict(groups), [a for a, v in groups.items() if len(v) == top], top


def majority_expected(correct: Sequence, groups: dict, tied: list) -> float:
    """Expected accuracy of the vote with ties broken uniformly at random."""
    if not tied:
        return 0.0
    return float(np.mean([bool(correct[groups[a][0]]) for a in tied]))


def tie_break(correct: Sequence, quality: Sequence[float], groups: dict,
              tied: list) -> float:
    """The vote, with the scorer choosing among answers tied at the top count.

    Where the plurality is unique this is the vote, unchanged: every candidate
    in the winning group carries the same answer, so no selector can move the
    outcome. §20.2's audit found that the hard way.
    """
    if not tied:
        return 0.0
    best = max(tied, key=lambda a: max(quality[i] for i in groups[a]))
    return float(bool(correct[groups[best][0]]))


def rerank(correct: Sequence, quality: Sequence[float],
           drawn: Sequence[int]) -> float:
    """Best-of-N by the scorer alone, ignoring the vote entirely.

    `len(drawn) == 0` rather than `not drawn`: the caller passes a numpy slice,
    and truthiness on an array of more than one element raises.
    """
    if len(drawn) == 0:
        return 0.0
    return float(bool(correct[max(drawn, key=lambda i: quality[i])]))


def oracle(correct: Sequence, drawn: Sequence[int]) -> float:
    return float(any(bool(correct[i]) for i in drawn))


def tokens_spent(tokens: Sequence[int], drawn: Sequence[int]) -> int:
    """Generation tokens paid for. Every drawn candidate is paid for in full."""
    return int(sum(tokens[i] for i in drawn))


def n_scored_lazy(groups: dict, tied: list) -> int:
    """Candidates a tie-break rule actually has to score.

    A tie-break consults the scorer only when the vote cannot decide, so its
    cost is the size of the tied bloc rather than N. §20.14 measured this at 1.5
    to 1.9 candidates per problem, which is the whole reason the rule is cheap,
    and it has to be reported rather than assumed.
    """
    return sum(len(groups[a]) for a in tied) if len(tied) > 1 else 0


def simulate_problem(answers: Sequence, correct: Sequence, tokens: Sequence[int],
                     qualities: dict[str, np.ndarray], ns: Sequence[int],
                     n_orders: int, rng: np.random.Generator) -> list[dict]:
    """Replay one problem's candidates in random draw orders.

    Returns one row per (order, N), carrying every rule's outcome on the same
    draws so all contrasts are paired. Averaging over orders is the caller's
    job, and doing it there rather than here is what keeps order-to-order
    variance from being mistaken for problem-to-problem variance.
    """
    m = len(correct)
    out: list[dict] = []
    for o in range(n_orders):
        order = rng.permutation(m)
        for n in ns:
            if n > m:
                continue
            drawn = order[:n]
            groups, tied, _ = vote(answers, drawn)
            row = {
                "order": o, "n": int(n),
                "tokens": tokens_spent(tokens, drawn),
                "tied": len(tied) > 1,
                "n_scored_lazy": n_scored_lazy(groups, tied),
                "majority": majority_expected(correct, groups, tied),
                "oracle": oracle(correct, drawn),
                "pass1": float(bool(correct[drawn[0]])),
            }
            for name, q in qualities.items():
                row[f"tiebreak::{name}"] = tie_break(correct, q, groups, tied)
                row[f"rerank::{name}"] = rerank(correct, q, drawn)
            out.append(row)
    return out
