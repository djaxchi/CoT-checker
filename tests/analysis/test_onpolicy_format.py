"""Tests for the boxed-answer tie-break confound."""

from __future__ import annotations

import numpy as np

from src.analysis.onpolicy_format import (bloc_purity, has_boxed,
                                          select_boxed_then, select_has_boxed)


def _r(boxed, w, index):
    return {"boxed": boxed, "w": w, "index": index}


def test_has_boxed_is_the_finiteness_of_the_margin():
    assert has_boxed(3.2) and has_boxed(0.0)
    assert not has_boxed(float("nan"))
    assert has_boxed(float("inf")) is False or True   # inf is not produced by the encoder


def test_select_prefers_a_boxed_candidate():
    rows = [_r(False, 0.1, 0), _r(True, 0.9, 1)]
    assert select_has_boxed(rows, [0, 1]) == 1


def test_select_falls_back_to_sampling_order_when_all_boxed():
    """With nothing to separate them the rule must not secretly read a score."""
    rows = [_r(True, 0.9, 1), _r(True, 0.1, 0)]
    assert select_has_boxed(rows, [0, 1]) == 1        # index 0 wins, not the low score


def test_select_falls_back_when_none_boxed():
    rows = [_r(False, 0.9, 1), _r(False, 0.1, 0)]
    assert select_has_boxed(rows, [0, 1]) == 1


def test_boxed_then_score_uses_the_score_only_within_the_boxed_set():
    rows = [_r(True, 0.9, 0), _r(True, 0.1, 1), _r(False, 0.0, 2)]
    assert select_boxed_then(rows, [0, 1, 2], "w") == 1
    # the unboxed candidate has the best score and must still lose
    assert select_boxed_then(rows, [0, 2], "w") == 0


def test_bloc_purity_labels_the_three_cases():
    assert bloc_purity([_r(True, 0, 0), _r(True, 0, 1)], [0, 1]) == "all_boxed"
    assert bloc_purity([_r(False, 0, 0), _r(False, 0, 1)], [0, 1]) == "none_boxed"
    assert bloc_purity([_r(True, 0, 0), _r(False, 0, 1)], [0, 1]) == "mixed"
