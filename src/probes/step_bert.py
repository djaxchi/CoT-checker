"""Step-level bidirectional encoder for step_bert_probe_v0.

One token per reasoning step (pooled frozen LLM states). A small pre-norm
transformer with ALiBi distance bias reads the trace; what each target step may
see is set by a per-sequence KEY mask, so information from hidden steps cannot
reach the target through any layer (a hidden step is never a key, at any depth).

Visibility modes for target step t:
    both   every step
    left   steps j <= t
    right  steps j >= t
    local  only t
and an optional window keeps |j - t| <= W on top of the mode.

Pretraining replaces the target token with a learned [MASK] embedding and
reconstructs a fixed (PCA-whitened) target vector from the visible context.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("both", "left", "right", "local")


def visibility(lengths: torch.Tensor, targets: torch.Tensor, modes: list[str] | str,
               T: int, window: int | None = None) -> torch.Tensor:
    """(B, T) bool, True = step j is a visible key for sequence b's target."""
    B = lengths.shape[0]
    if isinstance(modes, str):
        modes = [modes] * B
    j = torch.arange(T, device=lengths.device)[None, :]
    t = targets[:, None]
    vis = j < lengths[:, None]
    m = torch.tensor([MODES.index(x) for x in modes], device=lengths.device)[:, None]
    allowed = ((m == 0)
               | ((m == 1) & (j <= t))
               | ((m == 2) & (j >= t))
               | ((m == 3) & (j == t)))
    vis = vis & allowed
    if window is not None:
        vis = vis & ((j - t).abs() <= window)
    return vis


def alibi_slopes(n_heads: int) -> torch.Tensor:
    return torch.tensor([2 ** (-8.0 * (h + 1) / n_heads) for h in range(n_heads)])


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, ff: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self.p_attn = dropout

    def forward(self, h: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, T, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, T, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                           dropout_p=self.p_attn if self.training else 0.0)
        h = h + self.drop(self.out(a.transpose(1, 2).reshape(B, T, d)))
        return h + self.drop(self.ff(self.ln2(h)))


class StepBert(nn.Module):
    def __init__(self, n_parts: int, part_dim: int = 4096, d: int = 256, n_layers: int = 2,
                 n_heads: int = 4, ff: int = 1024, dropout: float = 0.1, n_target: int = 256):
        super().__init__()
        self.n_parts, self.part_dim, self.d = n_parts, part_dim, d
        self.part_norm = nn.LayerNorm(part_dim)
        self.proj = nn.Linear(n_parts * part_dim, d)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, n_heads, ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.recon = nn.Linear(d, n_target)
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.part_norm(x.view(B, T, self.n_parts, self.part_dim)).view(B, T, -1)
        return self.proj(x)

    def encode(self, x: torch.Tensor, key_vis: torch.Tensor,
               masked: torch.Tensor | None = None) -> torch.Tensor:
        """x (B,T,n_parts*part_dim), key_vis (B,T) bool, masked (B,T) bool -> (B,T,d)."""
        return self.encode_h(self.embed(x), key_vis, masked)

    def encode_h(self, h: torch.Tensor, key_vis: torch.Tensor,
                 masked: torch.Tensor | None = None) -> torch.Tensor:
        """Same as encode, from already-embedded tokens h (B,T,d)."""
        B, T, _ = h.shape
        if masked is not None:
            h = torch.where(masked[..., None], self.mask_emb.to(h.dtype), h)
        pos = torch.arange(T, device=h.device)
        dist = (pos[None, :] - pos[:, None]).abs().to(h.dtype)
        bias = -self.slopes.to(h.dtype)[:, None, None] * dist[None]            # (H,T,T)
        keymask = torch.zeros(B, 1, 1, T, device=h.device, dtype=h.dtype)
        keymask = keymask.masked_fill(~key_vis[:, None, None, :], -math.inf)
        bias = bias[None] + keymask                                           # (B,H,T,T)
        for blk in self.blocks:
            h = blk(h, bias)
        return self.norm(h)

    def reconstruct(self, x: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor,
                    modes: list[str] | str, extra_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Predicted target vector of step `targets[b]` with that step masked -> (B, n_target)."""
        return self.reconstruct_h(self.embed(x), lengths, targets, modes, extra_mask)

    def reconstruct_h(self, h: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor,
                      modes: list[str] | str, extra_mask: torch.Tensor | None = None) -> torch.Tensor:
        """reconstruct() from already-embedded tokens; embed once, score every target."""
        B, T, _ = h.shape
        vis = visibility(lengths, targets, modes, T)
        masked = F.one_hot(targets, T).bool()
        if extra_mask is not None:
            masked = masked | (extra_mask & vis)
        z = self.encode_h(h, vis, masked)
        return self.recon(z[torch.arange(B, device=h.device), targets])


class StepBertClassifier(nn.Module):
    """Fine-tuning head: MLP over [z_t, embed(x_t), extra scalar features]."""

    def __init__(self, encoder: StepBert, n_extra: int = 0, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.d
        self.head = nn.Sequential(nn.LayerNorm(2 * d + n_extra), nn.Linear(2 * d + n_extra, hidden),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor,
                modes: list[str] | str, window: int | None = None,
                extra: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        vis = visibility(lengths, targets, modes, T, window)
        z = self.encoder.encode(x, vis)
        idx = torch.arange(B, device=x.device)
        feats = [z[idx, targets], self.encoder.embed(x)[idx, targets]]
        if extra is not None:
            feats.append(extra)
        return self.head(torch.cat(feats, -1)).squeeze(-1)
