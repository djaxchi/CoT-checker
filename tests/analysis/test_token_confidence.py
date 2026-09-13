"""Tests for the DeepConf token-confidence family.

The equations are short enough that the risk is not arithmetic, it is silent
degeneracy: a window wider than the trace collapsing Eq 4 to Eq 3, a bottom-10%
of three groups rounding to zero groups and returning nan, a -inf logprob
poisoning a mean. Each of those has its own test because each would produce a
plausible-looking number rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import token_confidence as tc


def test_token_confidence_matches_equation_2():
    lp = np.log(np.array([[0.5, 0.25], [0.1, 0.05]]))
    got = tc.token_confidence(lp)
    want = -np.array([(np.log(0.5) + np.log(0.25)) / 2,
                      (np.log(0.1) + np.log(0.05)) / 2])
    assert np.allclose(got, want)


def test_token_confidence_rejects_wrong_rank():
    with pytest.raises(ValueError):
        tc.token_confidence(np.zeros(5))


def test_token_confidence_empty_is_empty_not_nan():
    assert tc.token_confidence(np.zeros((0, 4))).shape == (0,)


def test_group_confidence_is_a_stride_one_sliding_mean():
    c = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(tc.group_confidence(c, 2), [1.5, 2.5, 3.5])


def test_group_window_wider_than_trace_gives_one_group():
    """The paper's 1024/2048 windows exceed our ~470-token traces.

    The degenerate case must be one group equal to the trace mean, which makes
    Eq 5 and Eq 6 well defined, not an empty array that silently becomes nan.
    """
    c = np.array([1.0, 2.0, 3.0])
    g = tc.group_confidence(c, 1024)
    assert g.shape == (1,) and np.isclose(g[0], 2.0)
    assert np.isclose(tc.lowest_group(g), 2.0)
    assert np.isclose(tc.bottom_percent_group(g), 2.0)


def test_bottom_percent_always_takes_at_least_one_group():
    g = np.array([5.0, 1.0, 3.0])            # 10% of 3 floors to 0
    assert np.isclose(tc.bottom_percent_group(g, 10.0), 1.0)


def test_bottom_percent_averages_the_lowest_slice():
    g = np.arange(20, dtype=float)           # 10% of 20 -> 2 groups: 0 and 1
    assert np.isclose(tc.bottom_percent_group(g, 10.0), 0.5)


def test_tail_confidence_clips_to_trace_length():
    c = np.array([1.0, 2.0, 3.0])
    assert np.isclose(tc.tail_confidence(c, 2048), 2.0)
    assert np.isclose(tc.tail_confidence(c, 2), 2.5)


def test_step_min_uses_per_step_means_not_per_token_min():
    """The verifier's aggregation is min over steps of a per-step score.

    A per-token minimum would be a different rule and would beat or lose to the
    verifier for reasons that have nothing to do with the signal.
    """
    c = np.array([10.0, 0.0, 4.0, 4.0])
    spans = [(0, 2), (2, 4)]
    assert np.isclose(tc.step_min_confidence(c, spans), 4.0)


def test_step_min_ignores_empty_and_out_of_range_spans():
    c = np.array([1.0, 2.0])
    assert np.isclose(tc.step_min_confidence(c, [(0, 2), (2, 2), (5, 9)]), 1.5)


def test_step_min_with_no_usable_span_is_nan():
    assert np.isnan(tc.step_min_confidence(np.array([1.0]), [(3, 4)]))


def test_answer_token_margin_is_top1_minus_top2_in_nats():
    lp = np.log(np.array([[0.9, 0.05, 0.01], [0.6, 0.3, 0.05]]))
    got = tc.answer_token_margin(lp, (0, 2))
    want = ((np.log(0.9) - np.log(0.05)) + (np.log(0.6) - np.log(0.3))) / 2
    assert np.isclose(got, want)


def test_answer_token_margin_needs_two_columns():
    assert np.isnan(tc.answer_token_margin(np.zeros((3, 1)), (0, 3)))


def test_rule_family_is_callable_and_named_stably():
    rules = tc.trace_rules()
    c = np.linspace(1.0, 5.0, 50)
    spans = [(0, 25), (25, 50)]
    lp = np.log(np.full((50, 3), 0.3))
    for name, fn in rules.items():
        val = fn(c, spans, lp)
        assert np.isfinite(val), f"{name} produced {val}"
    assert "step_min_conf" in rules
    assert "bottom10_group_w1024" in rules      # the degenerate case is present


def test_infinite_logprob_propagates_rather_than_being_hidden():
    lp = np.array([[0.0, -np.inf]])
    assert np.isinf(tc.token_confidence(lp)[0])
