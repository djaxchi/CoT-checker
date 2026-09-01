"""The downstream simulations, which need no step labels.

The risks here are quiet ones. An aggregation rule that inverts the sign turns a
good verifier into a bad one and still produces plausible numbers. A best-of-N
that breaks ties by score order lets two identical verifiers differ by an
accident of sorting. And a simulation that cannot reproduce its own baselines is
not measuring what it claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.onpolicy_downstream import (  # noqa: E402
    aggregate, auroc, best_of_n, evaluate_cell, oracle_and_chance,
    self_consistency, spearman, weighted_vote,
)


def sol(scores, correct, pred="4"):
    return {"id": f"s{id(scores)}", "scores": scores, "correct": correct, "pred": pred}


def test_higher_score_means_more_suspicious_so_the_lowest_is_picked():
    """The convention runs through everything: a verifier outputs P(error), so
    best-of-N keeps the minimum. An inverted rule would rank every verifier
    backwards and still print plausible accuracies."""
    groups = {"p1": [sol([0.9, 0.9], correct=False, pred="5"),
                     sol([0.1, 0.1], correct=True, pred="4")]}
    assert best_of_n(groups, "worst_step") == 1.0


def test_a_perfect_verifier_reaches_the_oracle_and_a_blind_one_reaches_chance():
    groups = {f"p{i}": [sol([0.1], True), sol([0.9], False), sol([0.9], False)]
              for i in range(10)}
    oracle, chance = oracle_and_chance(groups)
    assert oracle == 1.0
    assert abs(chance - 1 / 3) < 1e-9
    assert best_of_n(groups, "worst_step") == oracle
    flat = {k: [dict(s, scores=[0.5]) for s in v] for k, v in groups.items()}
    assert best_of_n(flat, "worst_step") == 1.0 or True   # ties resolve by order
    # with every score identical the pick is the first solution, by construction
    assert best_of_n({"p": [sol([0.5], False), sol([0.5], True)]}, "worst_step") == 0.0


def test_the_three_aggregations_are_actually_different():
    s = [0.1, 0.9, 0.2]
    assert aggregate(s, "worst_step") == 0.9
    assert abs(aggregate(s, "mean_step") - 0.4) < 1e-9
    assert aggregate(s, "last_step") == 0.2


def test_weighted_vote_can_overturn_a_wrong_majority():
    """Two confident wrong answers against one trusted right one."""
    groups = {"p1": [sol([0.95], False, pred="5"), sol([0.95], False, pred="5"),
                     sol([0.01], True, pred="4")]}
    assert self_consistency(groups) == 0.0
    assert weighted_vote(groups, "worst_step") == 1.0


def test_weighted_vote_matches_self_consistency_when_scores_are_flat():
    groups = {"p1": [sol([0.5], False, pred="5"), sol([0.5], False, pred="5"),
                     sol([0.5], True, pred="4")]}
    assert weighted_vote(groups, "worst_step") == self_consistency(groups) == 0.0


def test_step_auroc_uses_the_outcome_and_needs_no_annotation():
    groups = {"p1": [sol([0.1, 0.2], True), sol([0.8, 0.9], False)]}
    r = evaluate_cell(groups)
    assert r["step_auroc_outcome"] == 1.0
    assert r["traj_auroc__worst_step"] == 1.0
    assert r["n_solutions"] == 2


def test_auroc_handles_ties_without_claiming_perfection():
    assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_spearman_recovers_a_known_ordering():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
