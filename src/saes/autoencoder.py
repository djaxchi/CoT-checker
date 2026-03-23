"""Sparse Autoencoder (sparse projector) used inside SSAE.

Maps a dense step embedding h_k ∈ ℝ^d to a sparse latent vector
h_c ∈ ℝ^(d * sparsity_factor) via a linear layer + ReLU activation.

Ported and documented from the SSAE reference implementation (Miaow-Lab/SSAE).
"""

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Linear encoder with a ReLU bottleneck that promotes sparse activations.

    Args:
        n_inputs: Dimensionality of the input embedding (d).
        n_latents: Dimensionality of the sparse latent space (c = d * sparsity_factor).
        sparsity_factor: Expansion ratio n_latents / n_inputs. Stored for reference.
        activation: Non-linearity applied after the linear projection (default: ReLU).
    """

    def __init__(
        self,
        n_inputs: int,
        n_latents: int,
        sparsity_factor: int = 1,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.sparsity_factor = sparsity_factor

        # Weight matrix (no bias — bias is learned separately for better init control)
        self.encoder = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a dense embedding to a sparse latent vector.

        Args:
            x: Last-token embedding of shape (batch, 1, n_inputs).

        Returns:
            latents: Sparse activations of shape (batch, 1, n_latents).
        """
        pre_act = self.encoder(x) + self.latent_bias  # (batch, 1, n_latents)
        latents = self.activation(pre_act)  # sparsity via ReLU
        return latents
