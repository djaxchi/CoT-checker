"""Unit tests for the per-token incorrectness-trajectory core."""

import math

import numpy as np
import pytest

from src.analysis.token_trajectory import (
    coincidence,
    per_token_certainty,
    probe_scores,
    spike_stats,
)


class TestPerTokenCertainty:
    def test_uniform_row_exact(self):
        # One token, vocab 3, all-equal logits -> uniform distribution.
        out = per_token_certainty(np.zeros((1, 3)), np.array([0]))
        ln3 = math.log(3)
        assert out["nll"][0] == pytest.approx(ln3)
        assert out["entropy"][0] == pytest.approx(ln3)
        assert out["p_top1"][0] == pytest.approx(1 / 3)
        assert out["p_realized"][0] == pytest.approx(1 / 3)
        assert out["logit_gap"][0] == pytest.approx(0.0)

    def test_confident_row(self):
        out = per_token_certainty(np.array([[20.0, 0.0, 0.0]]), np.array([0]))
        assert out["nll"][0] == pytest.approx(0.0, abs=1e-6)
        assert out["entropy"][0] == pytest.approx(0.0, abs=1e-6)
        assert out["p_top1"][0] == pytest.approx(1.0, abs=1e-6)
        assert out["logit_gap"][0] == pytest.approx(20.0)

    def test_surprised_when_realized_is_not_top(self):
        # Model is confident about token 0, but token 2 was realized -> high nll,
        # yet p_top1 (about token 0) stays high: nll != -log(p_top1).
        out = per_token_certainty(np.array([[20.0, 0.0, 0.0]]), np.array([2]))
        assert out["nll"][0] > 15.0
        assert out["p_top1"][0] == pytest.approx(1.0, abs=1e-6)
        assert out["p_realized"][0] < 1e-6

    def test_shapes_and_validation(self):
        out = per_token_certainty(np.random.randn(5, 7), np.arange(5) % 7)
        for k in ("nll", "entropy", "logit_gap", "p_top1", "p_realized"):
            assert out[k].shape == (5,)
        with pytest.raises(ValueError):
            per_token_certainty(np.zeros((2, 3)), np.array([0]))  # T mismatch


class TestProbeScores:
    def test_exact(self):
        H = np.array([[1.0, 2.0], [3.0, 4.0]])
        s = probe_scores(H, np.array([1.0, 1.0]), b=0.5)
        assert s.tolist() == [3.5, 7.5]

    def test_dim_mismatch(self):
        with pytest.raises(ValueError):
            probe_scores(np.zeros((2, 3)), np.array([1.0, 1.0]))


class TestSpikeStats:
    def test_single_spike(self):
        m = spike_stats([0.0, 0.0, 0.0, 5.0])
        assert m["n"] == 4
        assert m["peak"] == 5.0
        assert m["argmax_idx"] == 3
        assert m["argmax_frac"] == pytest.approx(1.0)
        assert m["prominence"] == pytest.approx(5.0)  # peak - median(=0)
        # peak sits (5 - 1.25)/std above the mean; std = sqrt(4.6875)
        assert m["peakiness"] == pytest.approx(3.75 / math.sqrt(4.6875), rel=1e-6)

    def test_plateau_is_not_peaky(self):
        peaky = spike_stats([0.0, 0.0, 0.0, 5.0])["peakiness"]
        flat = spike_stats([4.0, 5.0, 4.5, 5.0])["peakiness"]
        assert flat < peaky

    def test_single_token(self):
        m = spike_stats([2.0])
        assert m["argmax_frac"] == 0.0 and m["peakiness"] == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            spike_stats([])


class TestCoincidence:
    def test_perfectly_aligned(self):
        c = coincidence([0.0, 0.0, 5.0], [0.0, 0.0, 9.0])
        assert c["argmax_score"] == 2 and c["argmax_uncertainty"] == 2
        assert c["argmax_distance_frac"] == 0.0
        assert c["within_step_corr"] == pytest.approx(1.0)

    def test_misaligned_distance(self):
        c = coincidence([0.0, 5.0, 0.0], [0.0, 0.0, 5.0])
        assert c["argmax_score"] == 1 and c["argmax_uncertainty"] == 2
        assert c["argmax_distance_frac"] == pytest.approx(0.5)

    def test_constant_vector_corr_is_nan(self):
        c = coincidence([1.0, 1.0, 1.0], [0.0, 1.0, 2.0])
        assert math.isnan(c["within_step_corr"])
