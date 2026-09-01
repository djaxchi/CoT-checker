"""Suppressor diagnostic: a feature can carry nothing alone and a lot in company."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from suppressor_check import partial_corr  # noqa: E402


def test_partial_correlation_matches_the_textbook_formula():
    rng = np.random.default_rng(0)
    c = rng.normal(size=3000)
    a = c + rng.normal(size=3000)
    b = 0.5 * c + rng.normal(size=3000)
    rab, rac, rbc = (np.corrcoef(x, y)[0, 1] for x, y in ((a, b), (a, c), (b, c)))
    want = (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
    assert partial_corr(a, b, c) == pytest.approx(want, abs=1e-6)


def test_it_detects_a_textbook_suppressor():
    """The classic case: s is uncorrelated with y, but correlated with the noise
    in the predictor, so removing the predictor exposes a real relationship. This
    is exactly the shape the geometry block appears to have."""
    rng = np.random.default_rng(1)
    n = 20000
    y = rng.normal(size=n)
    nuisance = rng.normal(size=n)
    pred = y + 2.0 * nuisance                 # a noisy view of y
    s = nuisance                              # carries only the nuisance
    assert abs(np.corrcoef(s, y)[0, 1]) < 0.05          # nothing on its own
    assert abs(np.corrcoef(s, pred)[0, 1]) > 0.5        # but tied to the predictor
    assert abs(partial_corr(s, y, pred)) > 0.4          # and a lot once it is out


def test_an_ordinary_predictor_does_not_look_like_a_suppressor():
    """Guard against calling everything a suppressor: a feature that plainly
    correlates with the label must not show a much larger partial correlation."""
    rng = np.random.default_rng(2)
    n = 20000
    y = rng.normal(size=n)
    s = y + rng.normal(size=n)
    pred = y + rng.normal(size=n)
    assert abs(np.corrcoef(s, y)[0, 1]) > 0.5
    assert abs(partial_corr(s, y, pred)) < abs(np.corrcoef(s, y)[0, 1])


def test_a_constant_feature_returns_zero_rather_than_nan():
    rng = np.random.default_rng(3)
    assert partial_corr(np.ones(500), rng.normal(size=500), rng.normal(size=500)) == 0.0
