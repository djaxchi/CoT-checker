"""Tests for the tts_roster_v1 selection rules.

The failures that matter here are the quiet ones: a nan quality winning a
comparison, a tie-break being credited where the plurality was unique, or a
scorer's sign flipped so the rule picks the worst candidate and the curve simply
looks disappointing rather than wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.tts_frontier import (majority_expected, n_scored_lazy, oracle,
                                       quality_from, rerank, simulate_problem,
                                       tie_break, tokens_spent, vote)


def test_quality_from_flips_sign_for_suspicion_scores():
    """Verifier scores are suspicion; every rule here maximises quality."""
    assert np.allclose(quality_from([0.1, 0.9], higher_is_better=False), [-0.1, -0.9])
    assert np.allclose(quality_from([0.1, 0.9], higher_is_better=True), [0.1, 0.9])


def test_quality_from_sends_nan_to_minus_infinity():
    """A candidate whose statistic failed must lose, not sit mid-pack."""
    q = quality_from([float("nan"), 0.5], higher_is_better=True)
    assert q[0] == -np.inf and q[1] == 0.5


def test_vote_ignores_unparsed_answers_but_they_are_still_drawn():
    groups, tied, top = vote([None, "a", "a", None], [0, 1, 2, 3])
    assert set(groups) == {"a"} and tied == ["a"] and top == 2


def test_vote_with_nothing_parseable_is_empty_not_a_crash():
    groups, tied, top = vote([None, None], [0, 1])
    assert groups == {} and tied == [] and top == 0


def test_majority_is_the_expectation_over_tied_answers():
    """Two answers tied at one vote each, one right: expectation is 0.5."""
    groups, tied, _ = vote(["a", "b"], [0, 1])
    assert majority_expected([True, False], groups, tied) == 0.5


def test_tie_break_cannot_move_a_unique_plurality():
    """Every candidate in the winning group carries the same answer, so the
    outcome label cannot change however the scorer ranks them. §20.2 found this
    the hard way, having credited a gain that was really tie-breaking."""
    answers, correct = ["a", "a", "b"], [True, True, False]
    groups, tied, _ = vote(answers, [0, 1, 2])
    q = quality_from([0.0, 0.0, 99.0], higher_is_better=True)   # scorer loves "b"
    assert tie_break(correct, q, groups, tied) == 1.0


def test_tie_break_picks_the_scorers_answer_on_a_real_tie():
    answers, correct = ["a", "b"], [False, True]
    groups, tied, _ = vote(answers, [0, 1])
    assert tie_break(correct, quality_from([0.0, 1.0], True), groups, tied) == 1.0
    assert tie_break(correct, quality_from([1.0, 0.0], True), groups, tied) == 0.0


def test_rerank_ignores_the_vote():
    """Three votes for a wrong answer, one right; rerank can still take it."""
    correct = [False, False, False, True]
    q = quality_from([0.1, 0.1, 0.1, 0.9], higher_is_better=True)
    assert rerank(correct, q, [0, 1, 2, 3]) == 1.0


def test_n_scored_lazy_is_zero_when_the_vote_decides():
    groups, tied, _ = vote(["a", "a"], [0, 1])
    assert n_scored_lazy(groups, tied) == 0


def test_n_scored_lazy_is_zero_for_a_unique_plurality():
    """c wins outright, so the scorer is never consulted however close it was."""
    groups, tied, _ = vote(["a", "b", "c", "c"], [0, 1, 2, 3])
    assert n_scored_lazy(groups, tied) == 0


def test_n_scored_lazy_counts_only_the_tied_bloc():
    """The cost claim of §20.14 rests on this being the bloc, not N."""
    groups, tied, _ = vote(["a", "b", "c"], [0, 1, 2])
    assert n_scored_lazy(groups, tied) == 3        # three-way tie, all scored
    groups, tied, _ = vote(["a", "a", "b", "b", "c"], [0, 1, 2, 3, 4])
    assert n_scored_lazy(groups, tied) == 4        # a and b tie at 2; c is not scored


def test_tokens_are_paid_for_every_drawn_candidate():
    assert tokens_spent([10, 20, 30], [0, 2]) == 40


def test_oracle_is_any_correct_among_those_drawn():
    assert oracle([False, True, False], [0, 1]) == 1.0
    assert oracle([False, True, False], [0, 2]) == 0.0


def test_rerank_accepts_a_numpy_slice():
    """simulate_problem passes numpy slices; `not drawn` raises on those."""
    q = quality_from([0.1, 0.9], higher_is_better=True)
    assert rerank([False, True], q, np.array([0, 1])) == 1.0
    assert rerank([False, True], q, np.array([], dtype=int)) == 0.0


def test_simulate_returns_paired_rows_for_every_rule():
    rng = np.random.default_rng(0)
    rows = simulate_problem(
        answers=["a", "b", "a"], correct=[True, False, True], tokens=[10, 10, 10],
        qualities={"v": quality_from([0.9, 0.1, 0.9], True)},
        ns=[1, 2, 3], n_orders=4, rng=rng)
    assert len(rows) == 12
    for r in rows:
        assert {"majority", "oracle", "pass1", "tiebreak::v", "rerank::v"} <= set(r)
        assert r["tokens"] == 10 * r["n"]


def test_simulate_skips_an_n_larger_than_the_pool():
    rng = np.random.default_rng(0)
    rows = simulate_problem(["a"], [True], [5], {}, ns=[1, 4], n_orders=1, rng=rng)
    assert [r["n"] for r in rows] == [1]


def test_simulate_is_deterministic_given_the_generator():
    a = simulate_problem(["a", "b"], [True, False], [1, 1],
                         {"v": quality_from([1.0, 0.0], True)}, [2], 3,
                         np.random.default_rng(7))
    b = simulate_problem(["a", "b"], [True, False], [1, 1],
                         {"v": quality_from([1.0, 0.0], True)}, [2], 3,
                         np.random.default_rng(7))
    assert a == b


def test_bootstrap_requires_values_and_clusters_to_align():
    """Real data hit this: a problem with fewer than n gradeable candidates
    contributes no row at that budget, so a cluster list built from the full
    problem list indexes past the end of the value array."""
    import pytest
    from scripts.analysis.tts_build_frontier import bootstrap_mean
    vals = np.array([1.0, 0.0, 1.0])
    assert bootstrap_mean(vals, ["q1", "q2", "q3"])[0] == pytest.approx(2 / 3)
    with pytest.raises(IndexError):
        bootstrap_mean(vals, ["q1", "q2", "q3", "q4"])


# --- the weighted-vote family (sprint 8) -----------------------------------
#
# The claim these tests defend is that majority voting and rerank are the two
# ends of one dial. If that is wrong the sprint-8 comparison against the
# literature's aggregator is comparing against something else.

from src.analysis.tts_frontier import (filtered_vote, softmax_vote,  # noqa: E402
                                       softmax_weights, weighted_vote)


def test_softmax_vote_at_beta_zero_is_the_plain_majority():
    answers = ["a", "a", "b"]
    correct = [False, False, True]
    q = np.array([0.0, 0.1, 9.0])
    drawn = np.arange(3)
    groups, tied, _ = vote(answers, drawn)
    assert softmax_vote(correct, answers, q, drawn, 0.0) == \
        majority_expected(correct, groups, tied)


def test_softmax_vote_at_large_beta_is_rerank():
    """One candidate holds all the weight, so its answer wins outright."""
    answers = ["a", "a", "b"]
    correct = [False, False, True]
    q = np.array([0.0, 0.1, 9.0])
    drawn = np.arange(3)
    assert softmax_vote(correct, answers, q, drawn, 60.0) == \
        rerank(correct, q, drawn) == 1.0


def test_softmax_weights_give_a_failed_statistic_no_vote():
    w = softmax_weights(np.array([-np.inf, 1.0, 2.0]), np.arange(3), 1.0)
    assert w[0] == 0.0 and w[1] > 0 and w[2] > w[1]


def test_softmax_weights_fall_back_to_uniform_without_spread():
    """A single draw, or a pool of equal scores, must not divide by zero."""
    w = softmax_weights(np.array([0.5, 0.5]), np.arange(2), 4.0)
    assert w[0] == w[1] > 0
    assert softmax_weights(np.array([0.5]), np.arange(1), 4.0)[0] > 0


def test_weighted_vote_lets_two_weak_votes_lose_to_one_strong_one():
    answers = ["a", "a", "b"]
    correct = [False, False, True]
    assert weighted_vote(correct, answers, [0.1, 0.1, 0.9], np.arange(3)) == 1.0
    assert weighted_vote(correct, answers, [0.4, 0.4, 0.5], np.arange(3)) == 0.0


def test_weighted_vote_clamps_negative_and_nan_rewards_to_no_vote():
    answers = ["a", "b"]
    correct = [True, False]
    assert weighted_vote(correct, answers, [0.3, -5.0], np.arange(2)) == 1.0
    assert weighted_vote(correct, answers, [0.3, float("nan")], np.arange(2)) == 1.0


def test_weighted_vote_with_no_weight_anywhere_scores_zero_not_a_coin():
    answers = ["a", "b"]
    assert weighted_vote([True, False], answers, [0.0, 0.0], np.arange(2)) == 0.0


def test_filtered_vote_at_eta_zero_is_the_plain_majority():
    answers = ["a", "a", "b"]
    correct = [False, False, True]
    q = np.array([0.0, 0.1, 9.0])
    drawn = np.arange(3)
    groups, tied, _ = vote(answers, drawn)
    assert filtered_vote(correct, answers, q, drawn, 0.0) == \
        majority_expected(correct, groups, tied)


def test_filtered_vote_drops_the_low_scoring_bloc():
    """DeepConf-low keeps the top tenth; here that is the single best trace."""
    answers = ["a", "a", "b"]
    correct = [False, False, True]
    q = np.array([0.0, 0.1, 9.0])
    assert filtered_vote(correct, answers, q, np.arange(3), 0.9) == 1.0


def test_filtered_vote_always_keeps_at_least_one_candidate():
    """eta=1.0 must not empty the pool and score a structural zero."""
    answers = ["a"]
    assert filtered_vote([True], answers, np.array([0.5]), np.arange(1), 1.0) == 1.0


def test_filtered_vote_can_weight_the_survivors_too():
    """Filter keeps both, then the reward decides: the paper's variant."""
    answers = ["a", "b"]
    correct = [False, True]
    q = np.array([0.2, 0.1])
    unweighted = filtered_vote(correct, answers, q, np.arange(2), 0.0)
    weighted = filtered_vote(correct, answers, q, np.arange(2), 0.0,
                             reward=[0.1, 0.9])
    assert unweighted == 0.5 and weighted == 1.0


def test_simulate_problem_emits_the_new_rules_only_when_asked():
    rng = np.random.default_rng(0)
    answers, correct, tokens = ["a", "b"], [True, False], [10, 10]
    q = {"s": np.array([0.9, 0.1])}
    plain = simulate_problem(answers, correct, tokens, q, [2], 1, rng)[0]
    assert not [k for k in plain if k.startswith(("softvote", "filtvote", "wvote"))]
    rich = simulate_problem(answers, correct, tokens, q, [2], 1, rng,
                            rewards={"s": np.array([0.9, 0.1])},
                            betas=[0.0, 2.0], etas=[0.5])[0]
    for key in ("softvote0::s", "softvote2::s", "filtvote0.5::s",
                "wvote::s", "filtwvote0.5::s"):
        assert key in rich, key
