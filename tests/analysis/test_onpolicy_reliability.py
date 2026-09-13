"""Features and folds for the majority-reliability question.

The control that matters here is the vote-only model. Without it, a model that
reads the hidden-state scores and predicts the vote's correctness would look
informative while only rediscovering that agreement predicts correctness.
"""

from __future__ import annotations

import numpy as np

from src.analysis.onpolicy_reliability import (
    FEATURE_SETS,
    SCORE_FEATURES,
    f1_at,
    grouped_folds,
    length_features,
    problem_features,
    score_features,
    trivial_f1,
    vote_features,
)


def cand(answer, scores, n_chars=100, n_steps=None):
    return {"answer": answer, "scores": list(scores), "n_chars": n_chars,
            "n_steps": n_steps or len(scores)}


def test_a_unanimous_vote_has_full_margin_and_no_entropy():
    rows = [cand("2", [0.1])] * 4
    f = vote_features(rows)
    assert f["top_frac"] == 1.0 and f["margin"] == 1.0
    assert f["entropy"] == 0.0 and f["n_distinct_frac"] == 0.25


def test_a_tied_vote_has_zero_margin():
    """Zero margin is exactly the co-plurality case the tie-break analysis isolates."""
    rows = [cand("2", [0.1]), cand("2", [0.1]), cand("3", [0.1]), cand("3", [0.1])]
    assert vote_features(rows)["margin"] == 0.0
    assert vote_features(rows)["top_frac"] == 0.5


def test_an_unparsed_answer_is_counted_but_never_wins():
    rows = [cand("2", [0.1]), cand(None, [0.1]), cand(None, [0.1])]
    f = vote_features(rows)
    assert f["ungradeable_frac"] == 2 / 3
    assert f["top_frac"] == 1 / 3


def test_the_score_contrast_is_the_bloc_against_the_losers():
    rows = [cand("2", [0.2]), cand("2", [0.4]), cand("3", [0.9])]
    f = score_features(rows, [0, 1])
    assert abs(f["bloc_mean_score"] - 0.3) < 1e-9
    assert f["bloc_min_score"] == 0.2 and f["bloc_max_score"] == 0.4
    assert abs(f["bloc_minus_rest"] - (0.3 - 0.9)) < 1e-9


def test_a_bloc_holding_every_candidate_has_no_contrast_to_draw():
    rows = [cand("2", [0.2]), cand("2", [0.4])]
    assert score_features(rows, [0, 1])["bloc_minus_rest"] == 0.0


def test_an_empty_bloc_gives_zeros_rather_than_a_crash():
    rows = [cand(None, [0.5])]
    assert score_features(rows, []) == {k: 0.0 for k in SCORE_FEATURES}
    assert length_features(rows, []) == {"mean_chars": 0.0, "mean_steps": 0.0}


def test_every_named_feature_set_is_actually_produced():
    rows = [cand("2", [0.2], n_chars=10), cand("3", [0.4], n_chars=30)]
    f = problem_features(rows, [0])
    for name, keys in FEATURE_SETS.items():
        assert all(k in f for k in keys), name


def test_a_repeated_question_never_straddles_a_fold():
    """284 question texts hide behind the 300 evaluation ids; folding on ids would
    fit and test on the same question."""
    groups = [f"q{i // 3}" for i in range(30)]
    folds = grouped_folds(groups, n_folds=5, seed=0)
    assert sum(len(f) for f in folds) == 30
    seen = {}
    for k, f in enumerate(folds):
        for i in f:
            seen.setdefault(groups[i], k)
            assert seen[groups[i]] == k


def test_folds_are_deterministic_under_the_seed():
    groups = [f"q{i}" for i in range(20)]
    a = [f.tolist() for f in grouped_folds(groups, 4, seed=3)]
    b = [f.tolist() for f in grouped_folds(groups, 4, seed=3)]
    c = [f.tolist() for f in grouped_folds(groups, 4, seed=4)]
    assert a == b and a != c


def test_the_trivial_baseline_is_calling_every_vote_correct():
    y = np.array([1, 1, 1, 0])
    assert abs(trivial_f1(y) - 2 * 0.75 / 1.75) < 1e-12
    assert f1_at(y, np.zeros(4), 0.5) == 0.0
