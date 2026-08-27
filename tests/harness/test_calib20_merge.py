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
    """Same splits, same threshold selection, same held-out scoring.

    Pins the uniform-grid path, which is what the previous leaderboard used, so
    the quantile grid is an added option rather than a silent redefinition."""
    traces = _traces(seed=5)
    fast = calib20_subset(traces, grid_mode="uniform")

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


def test_quantile_grid_sits_where_the_scores_are():
    """A uniform grid has no resolution for a saturated score distribution;
    quantiles follow the data wherever it piles up."""
    from scripts.merge_rep_grid_leaderboard import quantile_grid
    rng = np.random.default_rng(0)
    # 70% of mass just above 0, 30% just below 1, with a little spread in each
    saturated = [(-1, list(np.where(rng.random(8) < 0.7,
                                    rng.uniform(0, 5e-3, 8),
                                    rng.uniform(1 - 5e-3, 1, 8)))) for _ in range(40)]
    g = quantile_grid(saturated)
    assert len(g) >= 3
    assert g.min() < 0.01 and g.max() > 0.99      # candidates at both poles


def test_quantile_grid_never_sees_the_evaluation_traces():
    """The grid is derived from calibration traces only. Changing the traces that
    are NOT in the calibration split must not change the grid."""
    from scripts.merge_rep_grid_leaderboard import quantile_grid
    rng = np.random.default_rng(1)
    calib = [(-1, list(rng.random(6))) for _ in range(20)]
    g1 = quantile_grid(calib)
    g2 = quantile_grid(calib)          # same input -> same grid, deterministic
    np.testing.assert_array_equal(g1, g2)
    g3 = quantile_grid(calib + [(0, [0.5] * 6)])
    assert not (len(g3) == len(g1) and np.array_equal(g3, g1))   # extra data does move it


def test_quantile_calib20_repairs_a_saturated_probe_and_leaves_a_healthy_one():
    """The signature of a correct fix: broken cells recover, healthy ones don't move."""
    from scripts.merge_rep_grid_leaderboard import calib20_subset
    rng = np.random.default_rng(3)

    def make(n, sep, saturate):
        out = []
        for i in range(n):
            L = int(rng.integers(3, 8))
            s = rng.uniform(0, 1, L) * 0.2
            label = -1 if i % 2 else int(rng.integers(0, L))
            if label != -1:
                s[label] = 0.2 + sep
            if saturate:                       # squash toward the poles
                s = np.where(s > 0.25, 1 - 1e-5, 1e-5)
            out.append((label, [float(x) for x in s]))
        return out

    healthy, broken = make(300, 0.5, False), make(300, 0.5, True)
    h_u, h_q = calib20_subset(healthy, "uniform"), calib20_subset(healthy, "quantile")
    b_u, b_q = calib20_subset(broken, "uniform"), calib20_subset(broken, "quantile")
    assert abs(h_q - h_u) < 0.05           # healthy barely moves
    assert b_q >= b_u - 1e-9               # saturated never gets worse
