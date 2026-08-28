"""Rescaling must shrink the numbers without changing what the probe can express.

The stored numbers swing by about +-22 on the layer we now read, five times more
than on the previous backbone, which pins half the scores to 0.0000 and 0.9999.
Dividing them down fixes that. What it must not do is change the ranking a linear
probe can produce, or leak evaluation data into the statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.harness import rescale as rs  # noqa: E402


def test_rescaled_numbers_sit_near_zero_and_swing_by_one():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 22.6, (5000, 64)).astype(np.float32)
    out = rs.apply(x, rs.fit(x))
    assert abs(out.mean()) < 0.05
    assert abs(out.std() - 1.0) < 0.05


def test_a_dead_position_does_not_become_infinite():
    """A position that never varies has zero swing; dividing by it would poison
    every vector."""
    x = np.random.default_rng(1).normal(0, 5, (500, 8)).astype(np.float32)
    x[:, 3] = 2.0                                  # constant
    out = rs.apply(x, rs.fit(x))
    assert np.isfinite(out).all()


def test_a_linear_probe_can_express_the_same_ranking():
    """Rescaling is a per-position divide, so a linear probe can undo it: the
    ordering of scores is unchanged if the weights are adjusted to match."""
    rng = np.random.default_rng(2)
    x = rng.normal(1.5, 20.0, (400, 16)).astype(np.float32)
    stats = rs.fit(x)
    w = rng.normal(0, 1, 16).astype(np.float32)
    raw = x @ w
    # the equivalent weights on the rescaled data
    w2 = w * stats["std"]
    resc = rs.apply(x, stats) @ w2 + float(stats["mean"] @ w)
    np.testing.assert_allclose(raw, resc, rtol=1e-3, atol=1e-2)


def test_sparse_codes_keep_their_zeros():
    """Centring a 99%-zero vector would make it 100% non-zero and the whole
    reason for sparse storage disappears, so sparse codes are only divided."""
    rng = np.random.default_rng(3)
    dense = np.zeros((300, 40), np.float32)
    for r in range(300):
        idx = rng.choice(40, 4, replace=False)
        dense[r, idx] = rng.uniform(0.5, 3.0, 4)
    stats = rs.fit(dense, center=False)
    assert stats["center"] is False
    np.testing.assert_allclose(stats["mean"], 0.0)

    idx = np.array([2, 7, 11])
    vals = np.array([1.0, 2.0, 3.0], np.float32)
    out = rs.apply_sparse(vals, idx, stats)
    np.testing.assert_allclose(out, vals / stats["std"][idx], rtol=1e-6)
    assert (out != 0).all()                        # non-zeros stay non-zero
    # and a zero entry is simply absent, so it cannot become non-zero
    assert len(out) == len(vals)


def test_statistics_come_from_training_rows_only():
    """Fitting on train and applying to test is the point; fitting on everything
    would leak the evaluation distribution into the probe's inputs."""
    rng = np.random.default_rng(4)
    train = rng.normal(0, 10, (2000, 8)).astype(np.float32)
    test = rng.normal(50, 1, (2000, 8)).astype(np.float32)   # very different
    stats = rs.fit(train)
    out = rs.apply(test, stats)
    # test data is NOT forced to zero-mean: it keeps its own shift, as it must
    assert out.mean() > 3.0


def test_saving_and_loading_round_trips(tmp_path):
    x = np.random.default_rng(5).normal(2, 7, (600, 12)).astype(np.float32)
    stats = rs.fit(x)
    rs.save(tmp_path / "s.npz", stats)
    back = rs.load(tmp_path / "s.npz")
    np.testing.assert_allclose(rs.apply(x, stats), rs.apply(x, back))
