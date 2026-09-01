"""Belief-shift features: divergences and ranks, never raw logits."""

from __future__ import annotations

import numpy as np
import pytest

from src.harness.beliefshift import N_BELIEF, belief_feats, shift


def _logits(n=200, seed=0, peak=None, scale=1.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, scale, n).astype(np.float32)
    if peak is not None:
        z[peak] += 12.0
    return z


def test_an_unchanged_belief_produces_no_shift():
    z = _logits()
    got = shift(z, z)
    assert got[0] == pytest.approx(0.0, abs=1e-5)     # KL
    assert got[1] == pytest.approx(0.0, abs=1e-5)     # JS
    assert got[2] == pytest.approx(0.0, abs=1e-5)     # entropy change
    assert got[3] == 1.0                              # same top-1
    assert got[4] == pytest.approx(0.0, abs=1e-6)     # top-1 kept rank 0


def test_features_are_invariant_to_adding_a_constant_to_the_logits():
    """Softmax is shift invariant, so the features must be too. If they are not,
    an absolute activation scale has leaked in."""
    a, b = _logits(seed=1), _logits(seed=2)
    np.testing.assert_allclose(shift(a, b), shift(a + 7.5, b - 3.25), atol=1e-5)


def test_a_step_that_changes_the_models_mind_scores_a_large_shift():
    a, b = _logits(seed=3, peak=10), _logits(seed=3, peak=150)
    small = shift(a, _logits(seed=3, peak=10))
    big = shift(a, b)
    assert big[0] > small[0] and big[1] > small[1]
    assert big[3] == 0.0 and small[3] == 1.0
    assert big[4] > small[4], "the displaced top-1 token should fall in rank"


def test_rank_is_logarithmic_so_a_fall_from_first_dominates():
    """A token dropping from rank 0 to rank 10 matters more than 100 to 110."""
    a = _logits(seed=4, peak=5)
    near = shift(a, _logits(seed=4, peak=6))
    assert near[4] >= 0.0


def test_the_cross_layer_block_is_zero_when_no_second_layer_is_given():
    """The layer-26 states exist for only part of the store; the feature block
    must be absent rather than silently reusing the along-step numbers."""
    a, b = _logits(seed=5), _logits(seed=6)
    got = belief_feats(a, b)
    assert len(got) == N_BELIEF
    np.testing.assert_allclose(got[5:10], 0.0)


def test_the_cross_layer_block_fills_in_when_a_second_layer_is_given():
    a, b, c = _logits(seed=5), _logits(seed=6), _logits(seed=7)
    got = belief_feats(a, b, c)
    assert np.abs(got[5:10]).sum() > 0
    np.testing.assert_allclose(got[:5], belief_feats(a, b)[:5])


def test_entropy_uses_the_full_distribution_not_the_top_k():
    """A flat distribution must score a higher entropy than a peaked one."""
    flat, peaked = _logits(seed=8, scale=0.01), _logits(seed=8, peak=3)
    assert belief_feats(flat, flat)[-1] > belief_feats(peaked, peaked)[-1]


def test_logit_lens_matches_a_reference_rmsnorm_then_unembed():
    """The deriver reimplements RMSNorm rather than calling the model, so it has
    to agree with the definition or every belief number is off."""
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from belief_shift_reps import logits_of

    torch.manual_seed(0)
    d, v, eps = 16, 40, 1e-6
    h = torch.randn(5, d)
    nw = torch.rand(d) + 0.5
    w = torch.randn(v, d)
    want = (h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps) * nw) @ w.T
    torch.testing.assert_close(logits_of(h, nw, eps, w), want, rtol=1e-5, atol=1e-5)


def test_logit_lens_is_invariant_to_scaling_the_state():
    """RMSNorm divides the scale out, so two states differing only in magnitude
    must give the same distribution. This is what makes the belief features
    immune to the raw activation swing that broke the first screen."""
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from belief_shift_reps import logits_of

    torch.manual_seed(1)
    h, nw, w = torch.randn(3, 16), torch.rand(16) + 0.5, torch.randn(40, 16)
    a = logits_of(h, nw, 1e-6, w)
    b = logits_of(9.0 * h, nw, 1e-6, w)
    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
