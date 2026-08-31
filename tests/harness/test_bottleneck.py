"""A bottleneck is only useful here if it keeps the low-variance correctness
direction that a reconstruction objective throws away.

These tests build data with exactly that structure -- a large nuisance direction
and a tiny discriminative one -- and check that each objective behaves as its
description claims, including the unsupervised one that is supposed to help
without ever seeing a label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.harness.bottleneck import Bottleneck, signal_share  # noqa: E402
from src.harness.rescale import fit_whiten  # noqa: E402

D, N = 32, 4000


def toy(seed=0, signal_scale=1.0):
    """Big nuisance variance, tiny class-carrying direction: the situation in the
    real data, where correctness is ~0.01% of variance."""
    rng = np.random.default_rng(seed)
    y = (rng.random(N) < 0.5).astype(np.float32)
    nuisance = rng.normal(0, 5.0, (N, D)).astype(np.float32)
    direction = np.zeros(D, np.float32); direction[7] = 1.0
    x = nuisance + signal_scale * (2 * y - 1)[:, None] * direction
    return torch.from_numpy(x), torch.from_numpy(y)


def train(model, x, y, steps=400, **kw):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        loss, parts = model.loss(x, y, **kw)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return parts


def test_signal_share_is_near_zero_without_a_signal():
    """It must not have a floor set by the code width: the variance-along-the-
    class-direction version scores ~1/d_code on a random code and looks
    informative when it is not."""
    x, y = toy(signal_scale=0.0)
    assert signal_share(x, y) < 1e-3
    assert signal_share(toy(signal_scale=3.0)[0], y) > signal_share(x, y)

    rng = np.random.default_rng(9)
    random_code = torch.from_numpy(rng.normal(0, 1, (N, 4)).astype(np.float32))
    assert signal_share(random_code, y) < 0.01      # would be ~0.25 under the old metric


def test_plain_reconstruction_does_not_concentrate_the_signal():
    """The real failure mode, reproduced: reconstruction alone carries no more
    correctness than the raw input (0.93x over 5 seeds, and it lands either side
    of 1.0), which is why the reconstruction SAE lost up to 0.195 F1. Averaged
    over seeds, because a single run of this is noisy."""
    x, y = toy()
    raw = signal_share(x, y)
    shares = []
    for seed in range(4):
        torch.manual_seed(seed)
        m = Bottleneck(D, d_code=4, objective="recon")
        train(m, x, y)
        shares.append(signal_share(m.encode(x), y))
    assert np.mean(shares) < raw * 2.0        # no concentration, unlike supervision


def test_supervised_objectives_raise_the_signal_share_above_the_raw_input():
    x, y = toy()
    raw = signal_share(x, y)
    torch.manual_seed(0)
    base = Bottleneck(D, 4, "recon"); train(base, x, y)
    mixed = Bottleneck(D, 4, "mixed"); train(mixed, x, y, beta=100.0)
    ib = Bottleneck(D, 4, "ib"); train(ib, x, y, beta=100.0)
    b, m, i = (signal_share(v.encode(x), y) for v in (base, mixed, ib))
    assert m > raw * 2, f"mixed {m:.5f} vs raw {raw:.5f}"   # supervision concentrates it
    assert m > b                                            # and beats reconstruction
    assert i > m                                            # IB compresses hardest


def test_mixed_beta_trades_reconstruction_for_detection():
    """The knob has to actually move both terms, or the sweep measures nothing."""
    x, y = toy()
    lo = train(Bottleneck(D, 4, "mixed"), x, y, beta=0.01)
    hi = train(Bottleneck(D, 4, "mixed"), x, y, beta=100.0)
    assert hi["bce"] < lo["bce"]        # more weight on detection -> better detection
    assert hi["recon"] > lo["recon"]    # paid for in reconstruction


def test_whitened_reconstruction_uses_the_whitening_matrix():
    x, y = toy()
    W = torch.from_numpy(fit_whiten(x.numpy())["W"])
    m = Bottleneck(D, d_code=4, objective="recon_white")
    plain, _ = m.loss(x, y)                       # no matrix -> plain reconstruction
    white, _ = m.loss(x, y, whiten=W)
    assert not torch.isclose(plain, white)
    assert torch.isfinite(white)


def test_ib_reports_a_kl_term_and_stays_finite():
    x, y = toy()
    m = Bottleneck(D, d_code=4, objective="ib")
    parts = train(m, x, y, beta=10.0, steps=100)
    assert "kl" in parts and "bce" in parts
    assert np.isfinite(parts["total"])


def test_encoding_is_deterministic_even_for_the_variational_form():
    """The grid must evaluate a fixed function, not a sample."""
    x, y = toy()
    m = Bottleneck(D, d_code=4, objective="ib").eval()
    torch.testing.assert_close(m.encode(x), m.encode(x))


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="objective must be"):
        Bottleneck(D, 4, objective="magic")
