"""The arithmetic behind the tie-break comparison.

The mistake this replaces was a reading one: "inside the majority bloc" was taken
to mean "among solutions that already agree", when the bloc is every solution at
the top answer count and can span several tied answers. These tests pin the two
cases apart so the distinction cannot quietly collapse again.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.onpolicy_tiebreak import (
    RANDOM,
    SELECTORS,
    answer_groups,
    cluster_bootstrap,
    problem_record,
    selector_accuracy,
    top_bloc,
)


def cand(answer, correct, scores, index, n_chars=100, n_steps=None):
    return {"answer": answer, "correct": correct, "scores": list(scores),
            "n_chars": n_chars, "n_steps": n_steps or len(scores), "index": index}


def test_a_unique_plurality_bloc_is_one_answer_and_cannot_change_the_outcome():
    rows = [cand("2", True, [0.9], 0), cand("2", True, [0.1], 1),
            cand("3", False, [0.0], 2)]
    rec = problem_record(rows, SELECTORS)
    assert rec["tied"] is False and rec["n_top_answers"] == 1
    assert set(rec["accuracy"].values()) == {1.0}


def test_a_coplurality_bloc_spans_distinct_answers():
    """66 of the 300 evaluation problems look like this, and every correctness
    change the verifier produced came from one of them."""
    rows = [cand("2", True, [0.8], 0), cand("2", True, [0.7], 1),
            cand("3", False, [0.1], 2), cand("3", False, [0.2], 3)]
    rec = problem_record(rows, SELECTORS)
    assert rec["tied"] is True and rec["n_top_answers"] == 2
    assert rec["bloc_size"] == 4
    assert rec["accuracy"][RANDOM] == 0.5
    # The verifier calls the wrong answer least suspicious and loses the tie.
    assert rec["accuracy"]["verifier_worst_step"] == 0.0


def test_unparsed_answers_are_not_a_group_of_their_own():
    rows = [cand("2", True, [0.1], 0), cand(None, False, [0.0], 1)]
    assert answer_groups(rows) == {"2": [0]}
    assert top_bloc(answer_groups(rows)) == ([0], ["2"])


def test_a_problem_where_nothing_parsed_has_an_empty_bloc():
    rows = [cand(None, False, [0.1], 0)]
    rec = problem_record(rows, SELECTORS)
    assert rec["bloc_size"] == 0 and rec["accuracy"][RANDOM] == 0.0


def test_random_is_scored_as_its_expectation_not_a_draw():
    rows = [cand("2", True, [0.5], 0), cand("3", False, [0.5], 1)]
    bloc = [0, 1]
    assert selector_accuracy(rows, bloc, RANDOM) == 0.5
    assert selector_accuracy(rows, bloc, RANDOM) == 0.5


def test_every_selector_breaks_its_own_ties_by_sampling_order():
    """Otherwise a selector with a flat key would get a free coin flip that the
    others do not, and look better than it is."""
    rows = [cand("3", False, [0.4], 0, n_chars=50),
            cand("2", True, [0.4], 1, n_chars=50)]
    for rule in SELECTORS:
        assert selector_accuracy(rows, [0, 1], rule) == 0.0, rule
        assert selector_accuracy(rows, [1, 0], rule) == 0.0, rule


def test_length_selectors_actually_read_length():
    rows = [cand("3", False, [0.9], 0, n_chars=200),
            cand("2", True, [0.1], 1, n_chars=50)]
    assert selector_accuracy(rows, [0, 1], "shortest") == 1.0
    assert selector_accuracy(rows, [0, 1], "longest") == 0.0


def test_a_mixed_label_answer_group_is_reported_not_silently_averaged():
    """Two identical normalized answers with different outcome labels means the
    grader or the normalizer disagrees with itself, which is a bug to chase, not
    a number to average."""
    rows = [cand("2", True, [0.1], 0), cand("2", False, [0.2], 1)]
    assert problem_record(rows, SELECTORS)["mixed_label_answer_groups"] == ["2"]


def test_the_bootstrap_resamples_questions_not_problem_ids():
    """One question duplicated across ids is one observation. Resampling ids would
    treat it as several and report an interval that is too narrow."""
    paired = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    same = cluster_bootstrap(paired, ["q0"] * 4 + ["q1"] * 4, draws=2000)
    ids = cluster_bootstrap(paired, [f"q{i}" for i in range(8)], draws=2000)
    assert same["n_questions"] == 2 and ids["n_questions"] == 8
    width = lambda r: r["ci95"][1] - r["ci95"][0]
    assert width(same) > width(ids)
    assert same["delta"] == ids["delta"] == 0.0


def test_the_bootstrap_reports_whether_the_interval_clears_zero():
    always = cluster_bootstrap(np.ones(40), [f"q{i}" for i in range(40)], draws=2000)
    assert always["delta"] == 1.0 and always["crosses_zero"] is False


def test_an_unknown_rule_is_an_error_not_a_zero():
    rows = [cand("2", True, [0.1], 0)]
    with pytest.raises(KeyError):
        selector_accuracy(rows, [0], "vibes")


# ---- confidence selectors -------------------------------------------------

def _row(answer, correct, conf, index):
    return {"answer": answer, "correct": correct, "scores": [0.5],
            "n_chars": 10, "n_steps": 1, "index": index, "conf": conf}


def test_confidence_selector_picks_by_orientation():
    from src.analysis.onpolicy_tiebreak import (SELECTORS,
                                                register_confidence_selectors)
    register_confidence_selectors(["c"])
    rows = [_row("a", False, {"c": 1.0}, 0), _row("b", True, {"c": 5.0}, 1)]
    assert SELECTORS["conf:c:high"](rows, [0, 1]) == 1
    assert SELECTORS["conf:c:low"](rows, [0, 1]) == 0


def test_confidence_nan_never_wins():
    """A failed statistic must lose, not win by comparing first."""
    from src.analysis.onpolicy_tiebreak import (SELECTORS,
                                                register_confidence_selectors)
    register_confidence_selectors(["c"])
    rows = [_row("a", False, {"c": float("nan")}, 0), _row("b", True, {"c": 1.0}, 1)]
    assert SELECTORS["conf:c:high"](rows, [0, 1]) == 1
    assert SELECTORS["conf:c:low"](rows, [0, 1]) == 1


def test_confidence_missing_rule_is_treated_as_nan():
    from src.analysis.onpolicy_tiebreak import (SELECTORS,
                                                register_confidence_selectors)
    register_confidence_selectors(["absent"])
    rows = [_row("a", False, {}, 0), _row("b", True, {"absent": 2.0}, 1)]
    assert SELECTORS["conf:absent:high"](rows, [0, 1]) == 1


def test_confidence_ties_fall_back_to_sampling_order():
    from src.analysis.onpolicy_tiebreak import (SELECTORS,
                                                register_confidence_selectors)
    register_confidence_selectors(["c"])
    rows = [_row("a", False, {"c": 3.0}, 1), _row("b", True, {"c": 3.0}, 0)]
    assert SELECTORS["conf:c:high"](rows, [0, 1]) == 1      # index 0 wins


def test_best_confidence_rule_takes_the_maximum():
    from src.analysis.onpolicy_tiebreak import best_confidence_rule
    summary = {"rules": {"conf:a:high": {"tie_accuracy": 0.2},
                         "conf:b:low": {"tie_accuracy": 0.4},
                         "shortest": {"tie_accuracy": 0.9}}}
    assert best_confidence_rule(summary, ["conf:a:high", "conf:b:low"]) == "conf:b:low"


def test_best_confidence_rule_none_when_absent():
    from src.analysis.onpolicy_tiebreak import best_confidence_rule
    assert best_confidence_rule({"rules": {}}, ["conf:a:high"]) is None
