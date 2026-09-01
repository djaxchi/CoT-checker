"""Ridge probe: closed form, no budget, and a lambda path that spans the two
rules the conicity study compared (whitened at small lambda, centroid at large)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ridge_screen import ridge_path  # noqa: E402


def _data(n=800, d=6, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.float32)
    x = rng.normal(size=(n, d))
    x[:, 0] += 1.5 * y                       # only feature 0 carries the label
    return x.astype(np.float32), y


def test_solution_matches_the_direct_solve():
    x, y = _data()
    t = np.where(y > 0.5, 1.0, -1.0)
    for lam in (1e-2, 1.0, 1e3):
        want = np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ t)
        np.testing.assert_allclose(ridge_path(x, y, [lam])[lam], want, rtol=1e-6, atol=1e-8)


def test_it_is_deterministic_unlike_the_sgd_screen():
    """The whole reason for this script: the same input must give the same
    ranking every time, with no seed and no epoch count involved."""
    x, y = _data()
    a = ridge_path(x, y, [1.0])[1.0]
    b = ridge_path(x, y, [1.0])[1.0]
    np.testing.assert_array_equal(a, b)


def test_large_lambda_approaches_the_mean_difference_direction():
    """As lambda grows the solution goes to X'y / lambda, which is the centroid
    rule the conicity study scored at 0.63. If this fails the path does not span
    what it claims to span."""
    x, y = _data()
    t = np.where(y > 0.5, 1.0, -1.0)
    w = ridge_path(x, y, [1e12])[1e12]
    ref = x.T @ t
    cos = float(w @ ref / (np.linalg.norm(w) * np.linalg.norm(ref)))
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_small_lambda_approaches_the_whitened_direction():
    """And at the other end, the LDA-style solution that whitens by the data
    covariance."""
    x, y = _data()
    t = np.where(y > 0.5, 1.0, -1.0)
    w = ridge_path(x, y, [1e-9])[1e-9]
    ref = np.linalg.solve(x.T @ x, x.T @ t)
    cos = float(w @ ref / (np.linalg.norm(w) * np.linalg.norm(ref)))
    assert cos == pytest.approx(1.0, abs=1e-5)


def test_the_two_ends_of_the_path_are_actually_different():
    """If whitening and the centroid rule gave the same direction there would be
    nothing to sweep. The conicity study measured 0.63 against 0.82."""
    x, y = _data(seed=3)
    x[:, 1] += 4.0 * x[:, 0]                 # correlate features so whitening bites
    lo = ridge_path(x, y, [1e-9])[1e-9]
    hi = ridge_path(x, y, [1e12])[1e12]
    cos = abs(float(lo @ hi / (np.linalg.norm(lo) * np.linalg.norm(hi))))
    assert cos < 0.95


def test_one_eigendecomposition_serves_the_whole_path():
    lams = [1e-2, 1.0, 1e2, 1e4]
    x, y = _data()
    got = ridge_path(x, y, lams)
    assert set(got) == set(lams)
    t = np.where(y > 0.5, 1.0, -1.0)
    for lam in lams:
        want = np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ t)
        np.testing.assert_allclose(got[lam], want, rtol=1e-6, atol=1e-8)


def test_paired_bootstrap_beats_unpaired_at_detecting_a_small_real_gain():
    """The search has been treating 0.01 differences as meaningful without ever
    pricing them. Representations are scored on identical rows, so their errors
    are correlated and the comparison must be paired: an unpaired reading of two
    overlapping intervals hides a difference that is present in every resample."""
    from ridge_screen import _bootstrap

    rng = np.random.default_rng(0)
    n = 1500
    y = (rng.random(n) < 0.2).astype(np.float32)
    base = y + rng.normal(0, 1.0, n)          # shared, correlated component
    rows = [
        {"name": "a", "val_selected_pb": 0.0, "_scores": {"s": (base, y)}},
        {"name": "b", "val_selected_pb": 0.0,
         "_scores": {"s": (base + 0.06 * y, y)}},   # strictly better on every row
    ]
    _bootstrap(rows, 300, "a")
    b = next(r for r in rows if r["name"] == "b")
    assert b["vs_ref_median"] > 0
    assert b["vs_ref_win_rate"] > 0.95, "a uniformly better score lost resamples"
    lo_a, hi_a = next(r for r in rows if r["name"] == "a")["ci95"]
    assert lo_a < hi_a
    assert hi_a > b["ci95"][0], "intervals should overlap; the pairing is the point"
