"""Text encoder classifier for step_bert_probe_v1 (Arm B).

A pretrained bidirectional encoder (ModernBERT-base) reads

    [CLS] problem [SEP] step_lo [SEP] step_lo+1 [SEP] ... step_hi-1 [SEP]

and each read step is scored from the [SEP] that closes it concatenated with the
mean of its own token outputs. What a step may see is set by which steps are in
the input, not by an attention mask, so it is exact at every layer:

    local    lo = t, hi = t + 1     (problem + step t)
    causal   lo = 0, hi = t + 1     (problem + steps up to t)
    bidir    lo = 0, hi = n         (whole trace; every step read in one pass)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_example(prob: list[int], steps: list[list[int]], lo: int, hi: int, cls: int, sep: int):
    """ids, and per included step k: (k, span start, span end, sep position)."""
    ids = [cls] + list(prob) + [sep]
    reads = []
    for k in range(lo, hi):
        a = len(ids)
        ids += steps[k]
        reads.append((k, a, len(ids), len(ids)))
        ids.append(sep)
    return ids, reads


def span_window(n: int, t: int, condition: str) -> tuple[int, int]:
    if condition == "local":
        return t, t + 1
    if condition == "causal":
        return 0, t + 1
    if condition == "bidir":
        return 0, n
    raise ValueError(condition)


class TextStepClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, input_ids, attention_mask, rb, ra, re, rs) -> torch.Tensor:
        """rb/ra/re/rs (R,): batch row, span start, span end (exclusive), sep position."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        cs = torch.cat([torch.zeros_like(h[:, :1]), h.cumsum(1)], 1)          # (B, L+1, d)
        n = (re - ra).clamp(min=1).float()[:, None]
        mean = (cs[rb, re] - cs[rb, ra]) / n
        mean = torch.where((re > ra)[:, None], mean, torch.zeros_like(mean))
        return self.head(torch.cat([h[rb, rs], mean], -1)).squeeze(-1)
