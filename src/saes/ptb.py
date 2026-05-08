"""Predictive Transition Bottleneck (PTB).

Objective: predict delta_h_k = h_{k+1} - h_k through a sparse bottleneck,
forcing the latent to encode what a reasoning step *does* to the trajectory
rather than what it *says*.

Architecture:
    Encoder : Linear(d, d) + bias + ReLU (or TopK) -> z_k
    Decoder : Linear(d, d) -> delta_h_hat_k

No language decoder. No LLM backbone. Pure activation-space model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PredictiveTransitionBottleneck"]


class PredictiveTransitionBottleneck(nn.Module):
    """Sparse bottleneck trained to predict h_{k+1} - h_k.

    Args:
        d: Hidden dimension. Default 896 (Qwen2.5-0.5B).
        k: If set, use TopK activation (keep exactly k dims). If None, use ReLU+L1.
    """

    def __init__(self, d: int = 896, k: Optional[int] = None) -> None:
        super().__init__()
        self.d = d
        self.k = k
        self.enc_weight = nn.Parameter(torch.empty(d, d))
        self.enc_bias   = nn.Parameter(torch.zeros(d))
        self.dec        = nn.Linear(d, d, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.enc_weight)
        nn.init.xavier_uniform_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)

    def encode(self, h_k: torch.Tensor) -> torch.Tensor:
        """(B, d) -> z_k (B, d), sparse via ReLU or TopK."""
        pre = h_k @ self.enc_weight.T + self.enc_bias
        if self.k is not None:
            # TopK: keep exactly k largest positive activations per sample
            z = torch.zeros_like(pre)
            vals, idx = pre.topk(self.k, dim=-1)
            z.scatter_(-1, idx, F.relu(vals))
            return z
        return F.relu(pre)

    def decode(self, z_k: torch.Tensor) -> torch.Tensor:
        """(B, d) -> delta_h_hat_k (B, d)."""
        return self.dec(z_k)

    def forward(self, h_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_k, delta_h_hat). Both (B, d)."""
        z_k = self.encode(h_k)
        return z_k, self.decode(z_k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str, step: int = 0, best_val_loss: float = float("inf"),
             extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":          self.state_dict(),
            "config":         {"d": self.d, "k": self.k},
            "step":           step,
            "best_val_loss":  best_val_loss,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(
        cls, path: str | os.PathLike, device: str = "cpu"
    ) -> "PredictiveTransitionBottleneck":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = cls(d=cfg["d"], k=cfg.get("k"))
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        return model
