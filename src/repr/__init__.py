"""Representation-shaping objectives for Sprint 2 (fork-based preference learning).

These objectives shape the *latent representation* (encoder output) during SAE/AE
training. They are auxiliary terms added on top of the reconstruction loss; the
heads they use are discarded after training so the saved artifact stays a plain
encoder (`representation.pt`), keeping the downstream ProcessBench pipeline intact.
"""

from src.repr.objectives import (
    enumerate_fork_pairs,
    ranking_loss,
    triplet_loss,
)

__all__ = ["ranking_loss", "triplet_loss", "enumerate_fork_pairs"]
