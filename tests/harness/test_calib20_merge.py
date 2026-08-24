"""calib-20 in the merge must be the same number the v1 rows were scored with.

The merge computes calib-20 vectorized, because the straightforward nested loop
over cells x subsets x splits x thresholds x traces does not finish. A faster
implementation that quietly disagrees with the old one would make the new
leaderboard incomparable to the old, which is the exact failure the rebuild is
meant to prevent, so these tests pin the two together on random data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.pb_threshold_calibration import (  # noqa: E402
    select_threshold, stratified_calib_split, trace_f1_pb,
)
from scripts.merge_rep_grid_leaderboard import (  # noqa: E402
    CALIB_GRID, CALIB_SIZE, CALIB_SPLITS, calib20_subset, f1_pb_from_preds,
    pred_matrix,
)


def _traces(n=90, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        L = int(rng.integers(3, 9))
        s = rng.uniform(0, 1, L)
        label = -1 if i % 3 == 0 else int(rng.integers(0, L))
        if label != -1:
            s[label] = rng.uniform(0.6, 1.0)
        out.append((label, s.tolist()))
    return out


def test_vectorised_f1_matches_the_reference_at_every_threshold():
    traces = _traces()
    preds, labels = pred_matrix(traces, CALIB_GRID)
    fast = f1_pb_from_preds(preds, labels)
    slow = np.array([trace_f1_pb(traces, float(t))[0] for t in CALIB_GRID])
    np.testing.assert_allclose(fast, slow, atol=1e-12)


def test_vectorised_threshold_choice_matches_the_reference():
    traces = _traces(seed=3)
    preds, labels = pred_matrix(traces, CALIB_GRID)
    fast_t = float(CALIB_GRID[int(np.argmax(f1_pb_from_preds(preds, labels)))])
    slow_t, _ = select_threshold(traces, CALIB_GRID)
    assert abs(fast_t - slow_t) < 1e-12


def test_calib20_matches_a_direct_reimplementation_of_the_reference_protocol():
    """Same splits, same threshold selection, same held-out scoring."""
    traces = _traces(seed=5)
    fast = calib20_subset(traces)

    evals = []
    for sd in range(CALIB_SPLITS):
        rng = np.random.default_rng(sd)
        calib, ev = stratified_calib_split(traces, CALIB_SIZE, rng)
        t_star, _ = select_threshold(calib, CALIB_GRID)
        evals.append(trace_f1_pb(ev, t_star)[0])
    slow = float(np.mean(evals))
    assert abs(fast - slow) < 1e-12


def test_a_trace_with_no_step_above_any_threshold_predicts_no_error():
    preds, labels = pred_matrix([(-1, [0.001, 0.002])], CALIB_GRID)
    assert (preds == -1).all()
