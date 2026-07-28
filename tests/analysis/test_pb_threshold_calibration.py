"""Tests for the ProcessBench threshold-calibration analysis."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SCRIPTS_ANALYSIS = Path(__file__).resolve().parents[2] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS_ANALYSIS))

from pb_threshold_calibration import (  # noqa: E402
    select_threshold,
    stratified_calib_split,
    trace_f1_pb,
)


def test_trace_f1_pb_perfect():
    traces = [(-1, [0.1, 0.2]), (1, [0.1, 0.9])]
    f1, acc_err, acc_cor = trace_f1_pb(traces, 0.5)
    assert math.isclose(f1, 1.0)
    assert acc_err == 1.0 and acc_cor == 1.0


def test_trace_f1_pb_threshold_too_high_misses_errors():
    # error step score 0.9 < 0.95 -> predicted all-correct -> error miss
    traces = [(-1, [0.1, 0.2]), (1, [0.1, 0.9])]
    f1, acc_err, acc_cor = trace_f1_pb(traces, 0.95)
    assert acc_err == 0.0
    assert acc_cor == 1.0
    assert f1 == 0.0  # harmonic mean with a zero component


def test_trace_f1_pb_wrong_position_is_miss():
    # first step over threshold is index 0, but true first error is index 1
    traces = [(1, [0.9, 0.1])]
    _, acc_err, _ = trace_f1_pb(traces, 0.5)
    assert acc_err == 0.0


def test_select_threshold_picks_max():
    traces = [(-1, [0.1, 0.2]), (1, [0.1, 0.9])]
    grid = np.array([0.5, 0.95])
    t, f1 = select_threshold(traces, grid)
    assert t == 0.5
    assert math.isclose(f1, 1.0)


def test_stratified_split_sizes_and_disjoint():
    traces = [(1, [0.5]) for _ in range(4)] + [(-1, [0.5]) for _ in range(6)]
    rng = np.random.default_rng(0)
    calib, ev = stratified_calib_split(traces, 5, rng)
    assert len(calib) == 5
    assert len(ev) == 5
    # both classes present in calib
    assert any(t[0] != -1 for t in calib) and any(t[0] == -1 for t in calib)
    # union == all, disjoint (by count, since traces are duplicated we check totals)
    assert len(calib) + len(ev) == len(traces)
    n_err = sum(1 for t in calib + ev if t[0] != -1)
    n_cor = sum(1 for t in calib + ev if t[0] == -1)
    assert n_err == 4 and n_cor == 6


def test_stratified_split_caps_at_available():
    traces = [(1, [0.5]) for _ in range(2)] + [(-1, [0.5]) for _ in range(2)]
    rng = np.random.default_rng(1)
    calib, ev = stratified_calib_split(traces, 4, rng)
    assert len(calib) == 4 and len(ev) == 0
