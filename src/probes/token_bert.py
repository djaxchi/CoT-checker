"""Token-level encoder over frozen LLM token states for step_bert_probe_v1 (Arm A).

The input is every CoT token's frozen state (each step's pre-step boundary token
followed by its span tokens, steps in order). A small pre-norm transformer with
ALiBi distance bias reads the tokens. What each token may attend to is set by a
(B, T, T) query-key mask built from per-token step ids:

    local    key in the query's own step
    causal   key position <= query position
    all      every valid token

Finite step windows (bidir_W) are NOT expressed as a mask: with 2 layers a
mask of |step(k) - step(q)| <= W would reach 2W steps through composition, so
those conditions crop the input to steps [t - W, t + W] around one target step
instead (see scripts/step_bert/finetune_tok.py). `local` and `causal` compose to
themselves, so a mask is exact at any depth for them.

Each step is read out by attention pooling over its span tokens (learned query)
concatenated with its boundary token's output, then an MLP gives one logit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.probes.step_bert import Block, alibi_slopes

MASKS = ("local", "causal", "all")


def token_mask(step_id: torch.Tensor, valid: torch.Tensor, kind: str) -> torch.Tensor:
    """(B, T, T) bool, True = query q may attend to key k. step_id (B, T) long, valid (B, T) bool."""
    T = step_id.shape[1]
    allowed = valid[:, None, :] & valid[:, :, None]
    if kind == "local":
        allowed = allowed & (step_id[:, :, None] == step_id[:, None, :])
    elif kind == "causal":
        pos = torch.arange(T, device=step_id.device)
        allowed = allowed & (pos[None, None, :] <= pos[None, :, None])
    elif kind != "all":
        raise ValueError(kind)
    # padded queries attend to themselves only, so softmax never sees an empty row
    eye = torch.eye(T, dtype=torch.bool, device=step_id.device)[None]
    return allowed | (eye & ~valid[:, :, None])


class TokenBert(nn.Module):
    def __init__(self, in_dim: int = 4096, d: int = 256, n_layers: int = 2, n_heads: int = 4,
                 ff: int = 1024, dropout: float = 0.1, hidden: int = 256):
        super().__init__()
        self.d = d
        self.in_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d)
        self.type_emb = nn.Embedding(2, d)            # 0 span token, 1 boundary token
        self.blocks = nn.ModuleList(Block(d, n_heads, ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.pool_q = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)

    def encode(self, x: torch.Tensor, is_boundary: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """x (B,T,in_dim), is_boundary (B,T) bool, allowed (B,T,T) bool -> (B,T,d)."""
        T = x.shape[1]
        h = self.proj(self.in_norm(x)) + self.type_emb(is_boundary.long())
        pos = torch.arange(T, device=x.device)
        dist = (pos[None, :] - pos[:, None]).abs().to(h.dtype)
        bias = -self.slopes.to(h.dtype)[None, :, None, None] * dist[None, None]       # (1,H,T,T)
        bias = bias.masked_fill(~allowed[:, None], -math.inf)                           # (B,H,T,T)
        for blk in self.blocks:
            h = blk(h, bias)
        return self.norm(h)

    def readout(self, z: torch.Tensor, step_id: torch.Tensor, is_boundary: torch.Tensor,
                valid: torch.Tensor, n_out: int) -> torch.Tensor:
        """Logit per step. step_id indexes a flat step list of length n_out (shared
        across the batch, so ids are global); tokens with step_id < 0 are ignored."""
        d = z.shape[-1]
        zf, sid = z.reshape(-1, d), step_id.reshape(-1)
        bnd, ok = is_boundary.reshape(-1), valid.reshape(-1) & (step_id.reshape(-1) >= 0)
        span = ok & ~bnd
        s = torch.where(span, self.pool_q(zf).squeeze(-1).float(), torch.zeros(1, device=z.device))
        idx = sid.clamp(min=0)
        smax = torch.full((n_out,), -math.inf, device=z.device).scatter_reduce(
            0, idx[span], s[span].detach(), "amax")
        smax = torch.where(torch.isfinite(smax), smax, torch.zeros_like(smax))
        # no inf - inf anywhere, so masked positions cannot leak NaN gradients
        e = torch.where(span, torch.exp(s - smax[idx]), torch.zeros_like(s))
        den = torch.zeros(n_out, device=z.device).index_add_(0, idx, e)
        pooled = torch.zeros(n_out, d, device=z.device).index_add_(0, idx, e[:, None] * zf.float())
        pooled = pooled / den.clamp(min=1e-12)[:, None]
        b_out = torch.zeros(n_out, d, device=z.device).index_add_(0, idx[ok & bnd], zf[ok & bnd].float())
        return self.head(torch.cat([pooled, b_out], -1)).squeeze(-1)
