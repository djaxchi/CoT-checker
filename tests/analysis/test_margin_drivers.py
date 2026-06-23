"""Unit tests for the per-step margin-driver helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "margin_drivers", ROOT / "scripts" / "inspect_margin_drivers.py"
)
md = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(md)


def test_auc_perfect_and_chance():
    label = np.r_[np.zeros(50), np.ones(50)]
    perfect = np.r_[np.zeros(50), np.ones(50)]      # incorrect always higher
    assert md.auc(perfect, label) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert abs(md.auc(rng.normal(size=100), label) - 0.5) < 0.15


def test_step_idx_of():
    assert md.step_idx_of("prm800k::p12::s3::7") == 7
    assert md.step_idx_of("prm800k::p1::s0::0") == 0
    assert md.step_idx_of("malformed") == -1


def test_residualize_kills_signal_only_when_it_is_the_feature():
    rng = np.random.default_rng(1)
    n = 400
    label = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    length = np.r_[rng.normal(10, 3, n // 2), rng.normal(13, 3, n // 2)]  # overlapping confound
    # (a) score that IS length-driven -> removing length collapses its AUC to ~0.5
    score_len = 2.0 * length + 0.01 * rng.normal(size=n)
    assert md.auc(score_len, label) > 0.6
    assert abs(md.auc(md.residualize(score_len, length[:, None]), label) - 0.5) < 0.1
    # (b) score that is an INDEPENDENT separating signal -> length removal barely hurts it
    indep = np.r_[rng.normal(0, 1, n // 2), rng.normal(3, 1, n // 2)]
    auc_raw = md.auc(indep, label)
    auc_res = md.auc(md.residualize(indep, length[:, None]), label)
    assert auc_res > auc_raw - 0.1


def test_partial_corr_controls_confound():
    rng = np.random.default_rng(2)
    n = 500
    z = rng.normal(size=n)                # common cause
    x = z + 0.1 * rng.normal(size=n)
    y = z + 0.1 * rng.normal(size=n)      # x,y correlated only through z
    assert md.partial_corr(y, x, None) > 0.8                      # raw corr high
    assert abs(md.partial_corr(y, x, z[:, None])) < 0.2           # vanishes given z
