"""Unit tests for parametric_retrieval_sae_decomp pure helpers."""

from __future__ import annotations

import numpy as np

from scripts.parametric_retrieval.prgd_sae_decomp import (
    donor_added,
    frac_in_topk,
    n_to_capture,
)


def test_frac_in_topk_basic():
    v = np.array([3.0, -1.0, 0.0, 0.0, 0.0])   # L1 mass = 4
    assert frac_in_topk(v, 1) == 0.75           # top |.| = 3
    assert frac_in_topk(v, 2) == 1.0
    assert frac_in_topk(v, 99) == 1.0


def test_frac_in_topk_zero_vector():
    assert frac_in_topk(np.zeros(5), 3) == 0.0


def test_n_to_capture_threshold():
    v = np.array([10.0, 1.0, 0.0])              # cumfrac: 0.909, 1.0
    assert n_to_capture(v, 0.9) == 1
    v2 = np.array([5.0, 5.0, 0.0])              # need both for 0.9
    assert n_to_capture(v2, 0.9) == 2
    assert n_to_capture(np.zeros(4), 0.9) == 0


def test_donor_added_picks_increased_features():
    fd = np.array([0.0, 2.0, 5.0, 0.0])
    fr = np.array([0.0, 0.0, 1.0, 3.0])         # df = [0, 2, 4, -3]
    got = donor_added(fd, fr, top=2)
    assert got == [2, 1]                         # feature 2 (+4) then 1 (+2)
    # feature 3 decreased -> never added
    assert 3 not in donor_added(fd, fr, top=4)


def test_donor_added_empty_when_no_increase():
    fd = np.array([0.0, 0.0])
    fr = np.array([1.0, 1.0])
    assert donor_added(fd, fr) == []
