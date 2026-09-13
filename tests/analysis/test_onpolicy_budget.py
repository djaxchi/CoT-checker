"""The budget simulation, where the easy mistakes are all about what is visible.

A stopping rule that can see a candidate it has not drawn is measuring nothing,
and an answer rule that quietly resolves ties with the verifier while the control
resolves them by chance would hand the verifier a win it did not earn. Both are
pinned here.
"""

from __future__ import annotations

import numpy as np

from src.analysis.onpolicy_budget import (
    ANSWER_CHEAP_TIE,
    ANSWER_MAJORITY,
    ANSWER_VERIFIER_TIE,
    accuracy_at_budget,
    apply_threshold,
    draw_state,
    features,
    frontier_envelope,
    simulate,
)


def test_the_state_reads_only_the_draws_taken_so_far():
    """The fourth candidate is correct and least suspicious; a rule stopping at
    t=2 must not be able to see it."""
    answers = ["2", "2", "3", "9"]
    correct = [False, False, False, True]
    quality = [0.0, 0.0, 0.0, 10.0]
    st = draw_state(answers, correct, quality, [0, 1, 2, 3], 2)
    assert st["t"] == 2
    assert st["score"][0] == 0.0  # best quality seen, not the best that exists
    assert st[ANSWER_MAJORITY] == 0.0


def test_a_tie_is_scored_as_a_coin_flip_for_the_majority_rule():
    answers, correct = ["2", "3"], [True, False]
    st = draw_state(answers, correct, [0.0, 0.0], [0, 1], 2)
    assert st[ANSWER_MAJORITY] == 0.5


def test_the_verifier_answer_rule_resolves_the_same_tie_with_the_score():
    answers, correct = ["2", "3"], [True, False]
    good = draw_state(answers, correct, [1.0, 0.0], [0, 1], 2)
    bad = draw_state(answers, correct, [0.0, 1.0], [0, 1], 2)
    assert good[ANSWER_VERIFIER_TIE] == 1.0
    assert bad[ANSWER_VERIFIER_TIE] == 0.0
    # and the control is unmoved by the score, which is the point of keeping the
    # two rules apart
    assert good[ANSWER_MAJORITY] == bad[ANSWER_MAJORITY] == 0.5


def test_a_decided_vote_ignores_the_verifier_entirely():
    answers, correct = ["2", "2", "3"], [True, True, False]
    st = draw_state(answers, correct, [0.0, 0.0, 99.0], [0, 1, 2], 3)
    assert st[ANSWER_MAJORITY] == st[ANSWER_VERIFIER_TIE] == 1.0


def test_an_unparsed_answer_is_paid_for_but_cannot_win():
    answers, correct = [None, "2"], [False, True]
    st = draw_state(answers, correct, [99.0, 0.0], [0, 1], 2)
    assert st[ANSWER_MAJORITY] == 1.0
    assert st["t"] == 2  # the wasted draw still counts against the budget


def test_a_draw_with_nothing_parsed_has_no_state():
    assert draw_state([None], [False], [0.0], [0], 1) is None


def test_vote_features_cannot_smuggle_the_score_in():
    st = draw_state(["2", "3"], [True, False], [5.0, -5.0], [0, 1], 2)
    assert features(st, "vote") == st["vote"]
    assert len(features(st, "both")) == len(st["vote"]) + len(st["score"])
    flipped = draw_state(["2", "3"], [True, False], [-5.0, 5.0], [0, 1], 2)
    assert features(st, "vote") == features(flipped, "vote")


def test_never_clearing_the_threshold_pays_the_whole_budget():
    probs = np.array([0.1, 0.2, 0.3])
    assert apply_threshold(probs, 0.9) == 2
    assert apply_threshold(probs, 0.15) == 1


def test_the_envelope_keeps_only_points_that_are_not_dominated():
    """A method that reaches 0.50 at cost 4 gets no credit for also reaching 0.45
    at cost 6."""
    env = frontier_envelope([(4.0, 0.50), (6.0, 0.45), (8.0, 0.55), (2.0, 0.30)])
    assert env == [(2.0, 0.30), (4.0, 0.50), (8.0, 0.55)]


def test_accuracy_at_a_budget_never_borrows_from_a_costlier_point():
    env = [(2.0, 0.30), (4.0, 0.50), (8.0, 0.55)]
    assert accuracy_at_budget(env, 3.9) == 0.30
    assert accuracy_at_budget(env, 4.0) == 0.50
    assert np.isnan(accuracy_at_budget(env, 1.0))


def test_simulate_replays_every_candidate_in_some_order():
    rng = np.random.default_rng(0)
    answers, correct, quality = ["2", "3", "2"], [True, False, True], [1.0, 0.0, 0.5]
    runs = simulate(answers, correct, quality, [0, 1, 2], 4, rng)
    assert len(runs) == 4 and all(len(r) == 3 for r in runs)
    for r in runs:
        assert r[-1][ANSWER_MAJORITY] == 1.0  # 2 beats 3 once all three are in
        assert [st["t"] for st in r] == [1, 2, 3]


def test_a_tie_is_flagged_with_the_candidates_a_lazy_verifier_would_score():
    """Running the verifier only when the vote ties is what makes it cheap, so
    the count of candidates it would touch is part of the state."""
    st = draw_state(["2", "2", "3", "3", "9"], [True] * 5, [0.0] * 5,
                    [0, 1, 2, 3, 4], 5)
    assert st["tied"] is True and st["n_scored_if_lazy"] == 4  # not the singleton
    decided = draw_state(["2", "2", "3"], [True] * 3, [0.0] * 3, [0, 1, 2], 3)
    assert decided["tied"] is False and decided["n_scored_if_lazy"] == 0


def test_the_free_tiebreak_control_is_scored_on_the_same_tie():
    """If a ruler breaks the tie as well as the verifier does, the verifier has
    shown nothing, so both rules see the identical tie."""
    answers, correct = ["2", "3"], [True, False]
    cheap = [1.0, 0.0]      # prefers candidate 0, which is right
    st = draw_state(answers, correct, [0.0, 1.0], [0, 1], 2, cheap=cheap)
    assert st[ANSWER_CHEAP_TIE] == 1.0     # the ruler gets it
    assert st[ANSWER_VERIFIER_TIE] == 0.0  # the verifier does not
    assert st[ANSWER_MAJORITY] == 0.5


def test_without_a_free_ordering_the_control_is_not_silently_random():
    st = draw_state(["2", "3"], [True, False], [1.0, 0.0], [0, 1], 2)
    assert st[ANSWER_CHEAP_TIE] == st[ANSWER_VERIFIER_TIE]
