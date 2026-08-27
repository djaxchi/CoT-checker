"""Each sparse SAE readout must equal its dense twin computed on the codes.

The point of the sparse mirror is that sae_X is exactly what step_X would be if
the learner saw SAE codes instead of raw states. If the sparse construction and
the dense definition drift apart, the whole "does sparsity help?" contrast
becomes a comparison of two different pooling rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "public_sae"))
from derive_sae_rep import WIDTH, pooled_codes  # noqa: E402

from src.harness.qwen_scope import TopKSAE  # noqa: E402

D_MODEL, D_SAE, K = 8, 32, 4


def _sae(seed=0):
    g = torch.Generator().manual_seed(seed)
    return TopKSAE(W_enc=torch.randn(D_SAE, D_MODEL, generator=g),
                   b_enc=torch.randn(D_SAE, generator=g) * 0.1,
                   W_dec=torch.randn(D_MODEL, D_SAE, generator=g),
                   b_dec=torch.randn(D_MODEL, generator=g) * 0.1, k=K)


def _dense(idx, val, width):
    out = np.zeros(D_SAE * width, dtype=np.float32)
    out[idx] = val
    return out


@pytest.mark.parametrize("readout", ["sae_last", "sae_mean", "sae_delta",
                                     "sae_stats", "sae_boundary_stats"])
def test_sparse_readout_matches_the_dense_definition(readout):
    sae = _sae()
    g = torch.Generator().manual_seed(1)
    span = torch.randn(6, D_MODEL, generator=g)
    boundary = torch.randn(1, D_MODEL, generator=g)

    idx, val = pooled_codes(sae, span, boundary, readout)
    got = _dense(idx, val, WIDTH[readout])

    codes = sae.encode(span)
    bcode = sae.encode(boundary)[0]
    if readout == "sae_last":
        want = sae.encode(span[-1:])[0].numpy()
    elif readout == "sae_mean":
        want = codes.mean(0).numpy()
    elif readout == "sae_delta":
        want = (codes[-1] - bcode).numpy()
    else:
        stats = [codes.mean(0), codes.max(0).values, codes.min(0).values,
                 codes.std(0), codes[-1]]
        blocks = ([bcode] + stats) if readout == "sae_boundary_stats" else stats
        want = torch.cat(blocks).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_last_and_mean_mirror_their_dense_twins_exactly():
    """sae_last is the code of the same row last_token reads; sae_mean pools the
    same rows step_mean pools."""
    sae = _sae(2)
    span = torch.randn(5, D_MODEL, generator=torch.Generator().manual_seed(3))
    b = torch.zeros(1, D_MODEL)
    i_last, v_last = pooled_codes(sae, span, b, "sae_last")
    np.testing.assert_allclose(_dense(i_last, v_last, 1),
                               sae.encode(span[-1:])[0].numpy(), rtol=1e-5, atol=1e-6)
    i_mean, v_mean = pooled_codes(sae, span, b, "sae_mean")
    np.testing.assert_allclose(_dense(i_mean, v_mean, 1),
                               sae.encode(span).mean(0).numpy(), rtol=1e-5, atol=1e-6)


def test_wide_readouts_stay_sparse():
    """sae_stats is 5x wide but must not become dense, or the storage argument
    for CSR collapses."""
    sae = _sae(4)
    span = torch.randn(40, D_MODEL, generator=torch.Generator().manual_seed(5))
    b = torch.randn(1, D_MODEL)
    idx, _ = pooled_codes(sae, span, b, "sae_stats")
    assert len(idx) < 5 * D_SAE          # strictly sparse
    assert len(np.unique(idx)) == len(idx)


def test_single_token_step_has_zero_std():
    """A one-token step has no variance; std must be 0, not NaN."""
    sae = _sae(6)
    span = torch.randn(1, D_MODEL, generator=torch.Generator().manual_seed(7))
    idx, val = pooled_codes(sae, span, torch.zeros(1, D_MODEL), "sae_stats")
    dense = _dense(idx, val, 5)
    assert np.isfinite(dense).all()
    np.testing.assert_allclose(dense[3 * D_SAE:4 * D_SAE], 0.0)   # the std block
