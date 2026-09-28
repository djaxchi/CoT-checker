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
                     n_orders: int, rng: np.random.Generator,
                     rewards: dict[str, np.ndarray] | None = None,
                     betas: Sequence[float] = (),
                     etas: Sequence[float] = ()) -> list[dict]:
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
                for b in betas:
                    row[f"softvote{b:g}::{name}"] = softmax_vote(
                        correct, answers, q, drawn, b)
                for e in etas:
                    row[f"filtvote{e:g}::{name}"] = filtered_vote(
                        correct, answers, q, drawn, e)
                r = (rewards or {}).get(name)
                if r is not None:
                    row[f"wvote::{name}"] = weighted_vote(
                        correct, answers, r, drawn)
                    for e in etas:
                        row[f"filtwvote{e:g}::{name}"] = filtered_vote(
                            correct, answers, q, drawn, e, reward=r)
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# The weighted-vote family (sprint 8).
#
# Sprint 7 compared three rules: count the answers, let the scorer break ties,
# or let the scorer choose outright. The test-time-scaling literature does not
# use any of those three as its headline aggregator. It uses a *weighted* vote:
#
#     answer* = argmax_a  sum_{i : answer_i = a}  w_i
#
# with w_i the verifier's reward for candidate i (Lightman et al. 2023 best-of-N
# with a PRM; Uesato et al. 2022), or the trace's confidence (DeepConf
# arXiv:2508.15260 Eq 8). Majority voting is that rule at w_i = 1, and rerank is
# that rule as the weight concentration goes to infinity, so the three sprint-7
# rules and the field's rule all sit on one dial. `softmax_vote` exposes the
# dial, which is the honest way to ask whether our operating point was the
# right one rather than the one we happened to implement.
#
# All of these break a tie in total weight the way `majority_expected` does, by
# expectation rather than by a coin, for the reason stated at the top of this
# module.


def _vote_expected(correct: Sequence, answers: Sequence,
                   subset: Sequence[int], weight: Sequence[float]) -> float:
    """Accuracy of the answer with the largest total weight, ties averaged.

    A candidate with zero weight is still drawn and still paid for, it just
    casts no vote. That is the weighted analogue of the nan convention in
    `quality_from`: a statistic that failed to compute must not decide.
    """
    totals: dict[object, float] = defaultdict(float)
    rep: dict[object, int] = {}
    for i in subset:
        a = answers[i]
        if a is None:
            continue
        totals[a] += float(weight[i])
        rep.setdefault(a, i)
    if not totals:
        return 0.0
    top = max(totals.values())
    if top <= 0.0:
        return 0.0
    tied = [a for a, t in totals.items() if t == top]
    return float(np.mean([bool(correct[rep[a]]) for a in tied]))


def weighted_vote(correct: Sequence, answers: Sequence,
                  reward: Sequence[float], drawn: Sequence[int]) -> float:
    """The literature's rule, with the scorer's own reward as the weight.

    `reward` must already be on a non-negative scale where larger is better,
    which is the caller's job because only the caller knows the scorer: a probe
    emitting a suspicion probability contributes 1 - suspicion, and a
    confidence statistic contributes itself.
    """
    if len(drawn) == 0:
        return 0.0
    w = np.asarray(reward, dtype=float)
    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    return _vote_expected(correct, answers, drawn, w)


def softmax_weights(quality: Sequence[float], drawn: Sequence[int],
                    beta: float) -> np.ndarray:
    """Weights over the drawn pool, tempered by `beta`, indexed like `quality`.

    Quality is standardised within the pool first, so `beta` means the same
    thing for a scorer emitting probabilities and one emitting log-probabilities
    and the dial can be swept once for the whole roster. beta = 0 gives a plain
    count; large beta puts all the weight on the pool's best candidate.

    A pool whose qualities are all equal (or a single draw) has no spread to
    standardise, and falls back to uniform weights rather than dividing by zero.
    """
    w = np.zeros(len(quality), dtype=float)
    idx = np.asarray(list(drawn), dtype=int)
    if idx.size == 0:
        return w
    q = np.asarray([quality[i] for i in idx], dtype=float)
    finite = np.isfinite(q)
    if not finite.any():
        return w
    z = np.zeros_like(q)
    sd = q[finite].std()
    if sd > 0:
        z[finite] = (q[finite] - q[finite].mean()) / sd
    z[~finite] = -np.inf
    e = np.exp(beta * z - np.max(beta * z[finite]))
    e[~finite] = 0.0
    w[idx] = e
    return w


def softmax_vote(correct: Sequence, answers: Sequence,
                 quality: Sequence[float], drawn: Sequence[int],
                 beta: float) -> float:
    """The vote with weights tempered by `beta`. beta=0 is `majority_expected`."""
    if len(drawn) == 0:
        return 0.0
    return _vote_expected(correct, answers, drawn,
                          softmax_weights(quality, drawn, beta))


def filtered_vote(correct: Sequence, answers: Sequence,
                  quality: Sequence[float], drawn: Sequence[int],
                  eta: float, reward: Sequence[float] | None = None) -> float:
    """DeepConf's rule: drop the worst `eta` of the pool, then vote.

    DeepConf reports two settings, keeping the top 10% of traces and the top
    90% (arXiv:2508.15260 §3.2), so eta is 0.9 and 0.1 respectively. eta = 0 is
    a plain vote over everything drawn. At least one candidate always survives,
    because a rule that can discard the entire pool is a rule that scores zero
    on small budgets for reasons that have nothing to do with the scorer.

    Passing `reward` votes with the confidence-weighted variant among the
    survivors, which is what the paper actually reports; leaving it None counts
    the survivors equally, which isolates what the filter alone contributes.
    """
    idx = np.asarray(list(drawn), dtype=int)
    if idx.size == 0:
        return 0.0
    keep = max(1, int(np.ceil((1.0 - eta) * idx.size)))
    q = np.asarray([quality[i] for i in idx], dtype=float)
    order = np.argsort(-np.where(np.isfinite(q), q, -np.inf), kind="stable")
    survivors = idx[order[:keep]]
    if reward is None:
        w = np.ones(len(quality), dtype=float)
    else:
        r = np.asarray(reward, dtype=float)
        w = np.where(np.isfinite(r), np.maximum(r, 0.0), 0.0)
    return _vote_expected(correct, answers, survivors, w)
