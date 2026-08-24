"""The learner axis of the representation x learner grid.

The point of the grid is that a leaderboard row is a *pair*: a representation
(what vector or sequence the learner is shown) and a learner (what reads it).
The v1 leaderboard varied both at once, so "whole-step representations win" was
confounded with "whole-step rows happened to get the bigger detector". Here the
two axes are separated: every learner below is trained by one trainer, on one
split, under one hyperparameter protocol, so a cell differs from its neighbour in
exactly one coordinate.

Every learner exposes the same contract:

    forward(x, mask) -> (B,) logits

with x either (B, d) for a fixed-vector representation (mask ignored, pass None)
or (B, T, d) padded token states for a sequence representation with mask (B, T),
1 for a real token and 0 for padding. `build_learner` parses a spec string so
capacity can be swept without touching this file:

    linear                       one nn.Linear, d params
    mlp:h1024                    one hidden layer of width 1024
    mlp:h1024x2                  two hidden layers of width 1024
    attn_query                   one learned query pools the tokens, then linear
    transformer:d256,l2,ff1024,h4   the ReProbe-style encoder

`is_sequence(spec)` says which representations a learner can read, which is what
keeps the grid honest: fixed-vector reps take the vector learners, the token
sequence takes the sequence learners, and `step_mean x linear` is the bridge cell
that appears in both comparisons (mean-pooling the sequence then reading it with
a linear head is literally that pair), so the two halves share a y-axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

VECTOR_LEARNERS = ("linear", "mlp")
SEQUENCE_LEARNERS = ("attn_query", "transformer")


class LinearHead(nn.Module):
    """logit = w . x + b. The minimal readout; the v1 leaderboard's only learner."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class MLPHead(nn.Module):
    """ReLU MLP over a fixed vector: `depth` hidden layers of width `hidden`."""

    def __init__(self, in_dim: int, hidden: int = 1024, depth: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AttnQueryPool(nn.Module):
    """z = sum_i softmax(q . h_i) h_i ; logit = w . z + b.

    One learned query decides which of the step's tokens to read, at the smallest
    capacity that still makes the pooling rule learned rather than fixed.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.head = nn.Linear(d, 1)
        self.scale = d ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        att = (x @ self.q) * self.scale
        att = att.masked_fill(mask == 0, -1e9)
        w = torch.softmax(att, dim=-1).unsqueeze(-1)
        z = (w * x).sum(1)
        return self.head(z).squeeze(-1)


class TransformerPool(nn.Module):
    """proj(d->d_model) + learned pos -> N encoder layers -> masked mean -> head.

    The maximal learner on the sequence axis, and the ReProbe (Ni et al., 2025)
    detector design restricted to the last-layer states we store.
    """

    def __init__(self, d: int, d_model: int = 256, nhead: int = 4, nlayers: int = 2,
                 ff: int = 1024, t_max: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d_model)
        self.pos = nn.Parameter(torch.randn(t_max, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, 1)
        self.t_max = t_max

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if T > self.t_max:
            raise ValueError(f"sequence of {T} exceeds t_max={self.t_max}")
        h = self.proj(x) + self.pos[:T].unsqueeze(0)
        h = self.enc(h, src_key_padding_mask=(mask == 0))
        m = mask.unsqueeze(-1)
        z = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(z).squeeze(-1)


def parse_spec(spec: str) -> tuple[str, dict]:
    """'transformer:d256,l2' -> ('transformer', {'d': 256, 'l': 2})."""
    head, _, tail = spec.partition(":")
    head = head.strip()
    opts: dict[str, int | float] = {}
    for part in (p for p in tail.split(",") if p.strip()):
        part = part.strip()
        key = part[0]
        rest = part[1:]
        if head == "mlp" and key == "h" and "x" in rest:
            width, _, depth = rest.partition("x")
            opts["h"] = int(width)
            opts["depth"] = int(depth)
            continue
        opts[key] = float(rest) if "." in rest else int(rest)
    return head, opts


def is_sequence(spec: str) -> bool:
    """True if this learner consumes padded token sequences rather than vectors."""
    head, _ = parse_spec(spec)
    if head in SEQUENCE_LEARNERS:
        return True
    if head in VECTOR_LEARNERS:
        return False
    raise ValueError(f"unknown learner {spec!r}")


def build_learner(spec: str, in_dim: int, t_max: int = 512,
                  dropout: float = 0.1) -> nn.Module:
    """Instantiate a learner from its spec string. `in_dim` is the feature dim d."""
    head, o = parse_spec(spec)
    if head == "linear":
        return LinearHead(in_dim)
    if head == "mlp":
        return MLPHead(in_dim, hidden=int(o.get("h", 1024)),
                       depth=int(o.get("depth", 1)), dropout=dropout)
    if head == "attn_query":
        return AttnQueryPool(in_dim)
    if head == "transformer":
        return TransformerPool(
            in_dim, d_model=int(o.get("d", 256)), nhead=int(o.get("h", 4)),
            nlayers=int(o.get("l", 2)), ff=int(o.get("f", 1024)),
            t_max=t_max, dropout=dropout)
    raise ValueError(f"unknown learner {spec!r}")


def param_count(model: nn.Module) -> int:
    """Trainable parameters, the x-axis of the capacity curve the grid reports."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
