"""Unit tests for the Stage 0 confidence sidecar + battery helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


enc = _load("encode_fork_confidence", "encode_fork_confidence.py")
bat = _load("analyze_confidence_battery", "analyze_confidence_battery.py")


# --------------------------------------------------------------------------- #
# step_confidence pure math
# --------------------------------------------------------------------------- #

def test_step_confidence_known_values():
    # Two candidate tokens. Row 0: certain on token 1 (huge logit) -> nll~0, entropy~0,
    # big gap. Row 1: uniform over 3 -> nll=log3, entropy=log3, gap=0.
    pred = np.array([[0.0, 50.0, 0.0],
                     [1.0, 1.0, 1.0]])
    targets = np.array([1, 0])
    s = enc.step_confidence(pred, targets)
    assert s["n_step_tokens"] == 2
    assert s["nll_first"] == pytest.approx(0.0, abs=1e-6)      # certain & correct
    assert s["nll_last"] == pytest.approx(np.log(3), abs=1e-6)  # uniform
    assert s["entropy_mean"] == pytest.approx((0.0 + np.log(3)) / 2, abs=1e-6)
    assert s["logit_gap_mean"] == pytest.approx((50.0 + 0.0) / 2, abs=1e-6)
    assert s["nll_max"] == pytest.approx(np.log(3), abs=1e-6)


def test_step_confidence_nll_matches_log_softmax():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(5, 7))
    targets = rng.integers(0, 7, size=5)
    s = enc.step_confidence(pred, targets)
    logp = pred - np.log(np.exp(pred - pred.max(1, keepdims=True)).sum(1, keepdims=True)) \
        - pred.max(1, keepdims=True)
    expect = -logp[np.arange(5), targets].mean()
    assert s["nll_mean"] == pytest.approx(expect, abs=1e-9)


def test_step_confidence_empty_is_nan():
    s = enc.step_confidence(np.zeros((0, 4)), np.array([], dtype=int))
    assert s["n_step_tokens"] == 0
    assert np.isnan(s["nll_mean"]) and np.isnan(s["entropy_mean"])


def test_step_confidence_more_surprising_higher_nll():
    # A token the model dislikes should score higher NLL than one it likes.
    pred = np.array([[5.0, 0.0]])
    likely = enc.step_confidence(pred, np.array([0]))["nll_mean"]
    unlikely = enc.step_confidence(pred, np.array([1]))["nll_mean"]
    assert unlikely > likely


# --------------------------------------------------------------------------- #
# battery verdict logic
# --------------------------------------------------------------------------- #

def test_verdict_band_thresholds():
    assert "NOT reducible" in bat.verdict_band(0.1)
    assert "PARTIAL" in bat.verdict_band(0.4)
    assert "largely a confidence" in bat.verdict_band(0.9)


def test_battery_flags_surprise_when_score_is_nll():
    # Construct a score that IS the confidence feature -> removing it must collapse AUC,
    # and the verdict must say the probe is a surprise meter.
    rng = np.random.default_rng(1)
    n = 600
    label = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    nll = np.r_[rng.normal(0, 1, n // 2), rng.normal(2, 1, n // 2)]  # incorrect more surprising
    score = nll.copy()                                              # probe == surprise
    conf = {k: nll.copy() for k in bat.SUBSUME_FEATURES}
    m = bat.run_battery(score, label, conf)
    assert m["auc_probe_raw"] > 0.7
    assert m["frac_probe_lift_removed_by_confidence"] > 0.6
    assert "confidence/surprise meter" in m["verdict"]


def test_battery_flags_independent_signal_as_not_surprise():
    # A correctness signal orthogonal to a (still label-correlated) surprise feature must
    # survive residualization -> verdict NOT reducible.
    rng = np.random.default_rng(2)
    n = 600
    label = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    nll = np.r_[rng.normal(0, 1, n // 2), rng.normal(1.0, 1, n // 2)]   # weak surprise signal
    indep = np.r_[rng.normal(0, 1, n // 2), rng.normal(4.0, 1, n // 2)]  # strong correctness axis
    score = indep + 0.01 * nll          # mostly the independent separator
    conf = {k: nll.copy() for k in bat.SUBSUME_FEATURES}
    conf["nll_first"] = nll.copy(); conf["nll_last"] = nll.copy()
    m = bat.run_battery(score, label, conf)
    assert m["auc_probe_raw"] > 0.8
    assert m["frac_probe_lift_removed_by_confidence"] < 0.25
    assert "NOT reducible" in m["verdict"]
