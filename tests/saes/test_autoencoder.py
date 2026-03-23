"""Unit tests for SparseAutoencoder.

All tests use only CPU and no real model weights — fast and self-contained.
"""

import torch
import pytest
from src.saes.autoencoder import SparseAutoencoder


@pytest.fixture
def sae():
    return SparseAutoencoder(n_inputs=16, n_latents=64, sparsity_factor=4)


def test_output_shape(sae):
    x = torch.randn(2, 1, 16)
    out = sae(x)
    assert out.shape == (2, 1, 64)


def test_output_nonnegative_relu(sae):
    """ReLU activation means latents should be >= 0."""
    x = torch.randn(8, 1, 16)
    out = sae(x)
    assert (out >= 0).all()


def test_sparsity_relu_kills_negatives():
    """Inputs that produce all-negative pre-activations → all-zero latents."""
    sae = SparseAutoencoder(n_inputs=4, n_latents=8)
    with torch.no_grad():
        sae.encoder.weight.fill_(1.0)
        sae.latent_bias.fill_(-1e6)  # large negative bias → all pre-acts negative
    x = torch.ones(1, 1, 4)
    out = sae(x)
    assert (out == 0).all()


def test_latent_bias_is_learnable(sae):
    assert sae.latent_bias.requires_grad


def test_encoder_weight_no_bias(sae):
    assert sae.encoder.bias is None


def test_custom_activation():
    sae = SparseAutoencoder(n_inputs=8, n_latents=16, activation=torch.nn.Sigmoid())
    x = torch.randn(1, 1, 8)
    out = sae(x)
    assert ((out > 0) & (out < 1)).all()


def test_batch_size_one(sae):
    x = torch.randn(1, 1, 16)
    out = sae(x)
    assert out.shape == (1, 1, 64)


def test_gradient_flows():
    sae = SparseAutoencoder(n_inputs=4, n_latents=8)
    x = torch.randn(2, 1, 4)
    out = sae(x)
    loss = out.sum()
    loss.backward()
    assert sae.encoder.weight.grad is not None
    assert sae.latent_bias.grad is not None
