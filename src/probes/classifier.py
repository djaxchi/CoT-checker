"""3-layer MLP probe for predicting step-level properties from SSAE latents.

Architecture matches the reference implementation (Miaow-Lab/SSAE/classifier/classifier.py):
    Linear(n_latents → hidden) → LayerNorm → ReLU → Dropout
    → Linear(hidden → hidden//2) → LayerNorm → ReLU → Dropout
    → Linear(hidden//2 → 1)
"""

from pathlib import Path

import torch
import torch.nn as nn


class StepCorrectnessClassifier(nn.Module):
    """Binary classifier: sparse latent vector h_c → step correctness logit.

    Args:
        input_dim: Dimensionality of the SSAE latent vector (n_latents).
        hidden_dim: Width of the two hidden layers (default 1024).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 896,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (not probabilities).

        Args:
            x: (batch, input_dim) float tensor of SSAE latents.

        Returns:
            (batch, 1) logits — apply sigmoid for probabilities.
        """
        return self.net(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions (0 or 1).

        Args:
            x: (batch, input_dim)
            threshold: decision boundary on sigmoid output.

        Returns:
            (batch,) int tensor of 0/1 predictions.
        """
        with torch.no_grad():
            probs = torch.sigmoid(self.forward(x)).squeeze(-1)
            return (probs >= threshold).long()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model": self.state_dict(),
                "config": {
                    "input_dim": self.net[0].in_features,
                    "hidden_dim": self.net[0].out_features,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "StepCorrectnessClassifier":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = cls(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"])
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        return model
