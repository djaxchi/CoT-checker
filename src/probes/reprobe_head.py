"""ReProbe's step head (Ni et al., arXiv 2511.06209), reimplemented for reprobe_comparison_v1.

Same modules and parameter names as `luh/heads/uncertainty_head_claim.py`, so the
released `weights.pth` (JingweiNi/ReProbe_Qwen3-8B_{DeepSeek,self}_anno) loads with
`load_state_dict` unchanged:

    proj         Linear(F, 2h), LayerNorm, GELU, Dropout, Linear(2h, h), LayerNorm, GELU
    entity_embedding  Embedding(2, h), added to every token: 1 on the step's tokens
    transformer_encoder  nn.TransformerEncoder(h, heads, ff 2048, GELU, dropout), n layers
    classifier   Linear(h, h), LayerNorm, GELU, Dropout, Linear(h, 1)

One copy of the trace per step attends over the trace's token features, and the
step's logit reads the mean of its own tokens' outputs. Features are the backbone's
final (post-norm) hidden states, shifted by one: the feature of token t is the state
at position t - 1, the state that produced it (`step_feature_index`).

`future`:
  "visible"  every real token is attended, including tokens after the step. This is
             what the released code computes: its `mask_future_tokens` does
             `src_key_pd &= cumulative_mask`, which keeps positions after the step
             attended, and evaluation runs one trace at a time, so nothing is masked.
  "masked"   tokens after the step's last token are ignored (a causal step score).
"""

from __future__ import annotations

import torch
from torch import nn

FUTURE = ("visible", "masked")


class ReProbeHead(nn.Module):
    def __init__(self, feature_dim: int = 4096, head_dim: int = 512, n_layers: int = 1,
                 n_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, head_dim * 2), nn.LayerNorm(head_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim * 2, head_dim), nn.LayerNorm(head_dim), nn.GELU())
        self.entity_embedding = nn.Embedding(2, head_dim)
        layer = nn.TransformerEncoderLayer(d_model=head_dim, nhead=n_heads, dropout=dropout,
                                           activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.LayerNorm(head_dim), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(head_dim, 1))

    def reprobe_init_(self) -> "ReProbeHead":
        """`reinitialize_weights` from ReProbe's train_luh/run_train_luh.py: xavier-uniform
        for every module weight with >= 2 dims, uniform(0, 1) for 1-dim weights (LayerNorm),
        zero biases. Applied module by module, as `nn.Module.apply` does there."""
        def re(module):
            w = getattr(module, "weight", None)
            if isinstance(w, torch.Tensor):
                if w.ndim >= 2:
                    nn.init.xavier_uniform_(w)
                else:
                    nn.init.uniform_(w)
            b = getattr(module, "bias", None)
            if isinstance(b, torch.Tensor):
                nn.init.zeros_(b)
        self.apply(re)
        return self

    def forward(self, feats: torch.Tensor, valid: torch.Tensor, claims: torch.Tensor,
                trace_of: torch.Tensor, future: str = "visible") -> torch.Tensor:
        """Step logits, (C,).

        feats   (B, T, F) token features of B traces (padded)
        valid   (B, T) bool, real feature positions
        claims  (C, T) {0, 1}, the token positions of each of the C steps
        trace_of (C,) long, the trace each step belongs to
        """
        if future not in FUTURE:
            raise ValueError(f"future must be one of {FUTURE}, got {future!r}")
        h = self.proj(feats)                                   # (B, T, h)
        cm = claims.long()
        x = h[trace_of] + self.entity_embedding(cm)            # (C, T, h)
        ignore = ~valid[trace_of]                              # True = not attended
        if future == "masked":
            last = torch.where(cm.bool(), torch.arange(cm.shape[1], device=cm.device)[None], -1).amax(1)
            ignore = ignore | (torch.arange(cm.shape[1], device=cm.device)[None] > last[:, None])
        x = self.transformer_encoder(x, src_key_padding_mask=ignore)
        w = cm.unsqueeze(-1).to(x.dtype)
        rep = (x * w).sum(1) / cm.sum(1).clamp(min=1).unsqueeze(-1).to(x.dtype)
        return self.classifier(rep).squeeze(-1)


def step_feature_index(context_len: int, aligned_token_ids: list[int]) -> list[int]:
    """Feature positions of a step whose reply-relative token ids are `aligned_token_ids`,
    for a sequence whose reply starts at `context_len`: ReProbe sets mask[context_len + id]
    and drops position 0 (`mask[1:]`), so token p is read at feature p - 1."""
    return [context_len + int(i) - 1 for i in aligned_token_ids]


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
