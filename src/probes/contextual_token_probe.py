"""Direct token-attention probe over frozen backbone states, one score per step.

bidirectional_token_probe_v1 (docs/bidirectional_token_probe_v1_plan.md, sections 7-8).

Rows. A trace with token states H (n_tokens, d) and step spans [s_j, e_j) becomes
probe rows:
  * prefix rows: tokens [0, s_0)                      owner -1, role PREFIX
  * per step j: a BOUNDARY row copying H[s_j - 1]     owner j,  role BOUNDARY
               (the pre-step state; source position s_j - 1 is kept)
  * per step j: its tokens [s_j, e_j)                 owner j,  role TOKEN
Separator tokens between steps are not probe rows; the separator before step j
only enters as step j's boundary copy, which by causality has seen steps < j.

Queries. One learned query per scored step j (owner j, role QUERY). Queries read
token/boundary keys but are never keys themselves, so they cannot exchange
labels or scores.

Visibility (applied in every layer, to token updates and query reads). Row owner
a, key owner b (prefix = -1):
  prefix rows             b == -1
  local                   b == -1 or b == a
  causal                  b == -1 or b <= a
  full                    any b
  future1                 b == -1 or b <= a + 1, AND the view is physically
                          cropped after step i+1 when scoring target i
A banded mask over an uncropped trace would let two layers relay step i+2 into
target i, so future1 builds one cropped view per target.

Output: logit of P(step incorrect). Validity is 1 - sigmoid(logit). Steps are
scored independently: no monotonicity, no absorbing error state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = ("local", "causal", "future1", "full")
ROLE_PREFIX, ROLE_BOUNDARY, ROLE_TOKEN, ROLE_QUERY = 0, 1, 2, 3
PREFIX_OWNER = -1


@dataclass
class TraceRows:
    src: np.ndarray    # (R,) token index into the trace's state array
    owner: np.ndarray  # (R,) -1 prefix, else step index
    role: np.ndarray   # (R,)
    pos: np.ndarray    # (R,) absolute source token position
    q_pos: np.ndarray  # (T,) query position: last token of step j


def trace_rows(step_starts, step_ends) -> TraceRows:
    s0 = int(step_starts[0])
    if s0 < 1:
        raise ValueError("need at least one prefix token before step 0")
    src = [np.arange(s0)]
    owner = [np.full(s0, PREFIX_OWNER)]
    role = [np.full(s0, ROLE_PREFIX)]
    for j, (s, e) in enumerate(zip(step_starts, step_ends)):
        s, e = int(s), int(e)
        if e <= s:
            raise ValueError(f"empty step {j}")
        src.append(np.arange(s - 1, e))  # boundary copy + step tokens
        owner.append(np.full(e - s + 1, j))
        role.append(np.array([ROLE_BOUNDARY] + [ROLE_TOKEN] * (e - s)))
    src_a = np.concatenate(src).astype(np.int64)
    return TraceRows(src=src_a, owner=np.concatenate(owner).astype(np.int64),
                     role=np.concatenate(role).astype(np.int64), pos=src_a.copy(),
                     q_pos=np.asarray(step_ends, dtype=np.int64) - 1)


def views_for(rows: TraceRows, n_steps: int, arm: str, targets) -> list[tuple[np.ndarray, list[int]]]:
    """[(row_selection, query_steps)]: one view for local/causal/full, one cropped
    view per target for future1."""
    targets = [int(t) for t in targets]
    if arm not in ARMS:
        raise ValueError(arm)
    if arm != "future1":
        return [(np.ones(len(rows.src), dtype=bool), targets)] if targets else []
    return [(rows.owner <= min(t + 1, n_steps - 1), [t]) for t in targets]


def allowed(q_owner: torch.Tensor, k_owner: torch.Tensor, arm: str) -> torch.Tensor:
    """Boolean visibility (..., Lq, Lk) from owners (prefix = -1). Padding is
    handled by the caller."""
    a = q_owner.unsqueeze(-1)
    b = k_owner.unsqueeze(-2)
    kp = b == PREFIX_OWNER
    if arm == "local":
        ok = kp | (b == a)
    elif arm == "causal":
        ok = kp | (b <= a)
    elif arm == "full":
        ok = torch.ones_like(kp)
    elif arm == "future1":
        ok = kp | (b <= a + 1)
    else:
        raise ValueError(arm)
    return torch.where(a == PREFIX_OWNER, kp, ok)


@dataclass
class Batch:
    feats: torch.Tensor     # (N, d) unique frozen states of the batch's traces
    row_idx: torch.Tensor   # (S, L) index into feats (0 at padding)
    row_owner: torch.Tensor  # (S, L)
    row_role: torch.Tensor  # (S, L)
    row_pos: torch.Tensor   # (S, L)
    row_valid: torch.Tensor  # (S, L) bool
    q_step: torch.Tensor    # (S, Q) step index (owner) of each query
    q_pos: torch.Tensor     # (S, Q)
    q_valid: torch.Tensor   # (S, Q) bool
    q_trace: torch.Tensor   # (S, Q) batch-local trace index of each query
    arm: str

    def to(self, device) -> "Batch":
        kw = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
              for k, v in self.__dict__.items()}
        return Batch(**kw)


def collate(traces: list[dict], arm: str, targets: list[list[int]] | None = None) -> Batch:
    """traces: dicts with 'H' (n_tokens, d) array/tensor, 'step_starts', 'step_ends'.
    targets: per trace the step indices to score (default: every step)."""
    feats, seqs = [], []
    off = 0
    for ti, tr in enumerate(traces):
        rows = trace_rows(tr["step_starts"], tr["step_ends"])
        T = len(tr["step_starts"])
        tg = list(range(T)) if targets is None else targets[ti]
        H = tr["H"]
        feats.append(torch.as_tensor(np.asarray(H)) if not torch.is_tensor(H) else H)
        for sel, qs in views_for(rows, T, arm, tg):
            seqs.append((ti, off, rows, sel, qs))
        off += H.shape[0]
    S = len(seqs)
    L = max((int(sel.sum()) for _, _, _, sel, _ in seqs), default=1)
    Q = max((len(qs) for *_, qs in seqs), default=1)
    row_idx = torch.zeros(S, L, dtype=torch.long)
    row_owner = torch.full((S, L), PREFIX_OWNER, dtype=torch.long)
    row_role = torch.zeros(S, L, dtype=torch.long)
    row_pos = torch.zeros(S, L, dtype=torch.long)
    row_valid = torch.zeros(S, L, dtype=torch.bool)
    q_step = torch.zeros(S, Q, dtype=torch.long)
    q_pos = torch.zeros(S, Q, dtype=torch.long)
    q_valid = torch.zeros(S, Q, dtype=torch.bool)
    q_trace = torch.full((S, Q), -1, dtype=torch.long)
    for s, (ti, o, rows, sel, qs) in enumerate(seqs):
        n = int(sel.sum())
        row_idx[s, :n] = torch.from_numpy(rows.src[sel] + o)
        row_owner[s, :n] = torch.from_numpy(rows.owner[sel])
        row_role[s, :n] = torch.from_numpy(rows.role[sel])
        row_pos[s, :n] = torch.from_numpy(rows.pos[sel])
        row_valid[s, :n] = True
        m = len(qs)
        q_step[s, :m] = torch.tensor(qs)
        q_pos[s, :m] = torch.from_numpy(rows.q_pos[qs])
        q_valid[s, :m] = True
        q_trace[s, :m] = ti
    return Batch(torch.cat(feats, 0) if feats else torch.zeros(0, 1), row_idx, row_owner,
                 row_role, row_pos, row_valid, q_step, q_pos, q_valid, q_trace, arm)


def sinusoid(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """Fixed absolute encoding; depends only on the position, never on trace length."""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=pos.device, dtype=torch.float32) / half)
    ang = pos.float().unsqueeze(-1) * freq
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class _Layer(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x: torch.Tensor, n_keys: int, mask: torch.Tensor) -> torch.Tensor:
        # x: (S, L+Q, d); keys are x[:, :n_keys]; mask (S, 1, L+Q, n_keys) bool
        S, M, d = x.shape
        h = self.ln1(x)
        q = self.q(h).view(S, M, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(h[:, :n_keys]).view(S, n_keys, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(S, M, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class ContextualTokenProbe(nn.Module):
    """Pre-norm transformer over projected frozen token states + per-step queries."""

    def __init__(self, d_in: int, d: int = 256, layers: int = 2, heads: int = 4,
                 ff: int = 1024, dropout: float = 0.1, max_steps: int = 512):
        super().__init__()
        self.d = d
        self.in_norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d)
        self.role = nn.Embedding(4, d)
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.layers = nn.ModuleList(_Layer(d, heads, ff, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.max_steps = max_steps

    def _step_enc(self, owner: torch.Tensor) -> torch.Tensor:
        # prefix rows get no step encoding (their role embedding identifies them)
        e = sinusoid(owner.clamp(min=0), self.d)
        return e * (owner >= 0).unsqueeze(-1)

    def forward(self, b: Batch) -> torch.Tensor:
        """Returns (S, Q) logits of P(incorrect); invalid query slots are 0."""
        z = self.proj(self.in_norm(b.feats.to(self.proj.weight.dtype)))
        x = z[b.row_idx]
        x = x + sinusoid(b.row_pos, self.d) + self._step_enc(b.row_owner) + self.role(b.row_role)
        S, L = b.row_idx.shape
        Q = b.q_step.shape[1]
        qx = (self.query + sinusoid(b.q_pos, self.d) + self._step_enc(b.q_step)
              + self.role.weight[ROLE_QUERY])
        x = torch.cat([x, qx], dim=1)
        owners = torch.cat([b.row_owner, b.q_step], dim=1)
        vis = allowed(owners, b.row_owner, b.arm) & b.row_valid.unsqueeze(1)
        # padded rows/queries: let them see key 0 (always a valid prefix row) so the
        # softmax stays finite; their outputs are never read.
        pad = ~torch.cat([b.row_valid, b.q_valid], dim=1)
        vis[:, :, 0] |= pad
        mask = vis.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, L, mask)
        logits = self.head(self.out_norm(x[:, L:])).squeeze(-1)
        return logits.masked_fill(~b.q_valid, 0.0)


def gather_scores(b: Batch, logits: torch.Tensor) -> list[tuple[int, int, float]]:
    """[(batch-local trace index, step, logit)] for every valid query."""
    v = b.q_valid
    return list(zip(b.q_trace[v].tolist(), b.q_step[v].tolist(), logits[v].float().tolist()))


def target_weights(n_labeled: list[int], n_traj: int) -> list[float]:
    """Per-target loss weight 1 / (labeled targets in its trajectory * trajectories
    in the effective batch). Summed over every target of an effective batch this
    is the mean over trajectories of the mean BCE over each one's labeled targets,
    however the targets are split into views or microbatches."""
    return [1.0 / (n * n_traj) for n in n_labeled]


def weighted_bce(b: Batch, logits: torch.Tensor, y: dict, w: dict) -> torch.Tensor:
    """Sum of w * BCE over valid queries that carry a label.

    y, w: {(batch-local trace index, step): label / weight}. Steps without a key
    (unknown labels) contribute nothing."""
    tr = b.q_trace[b.q_valid].tolist()
    st = b.q_step[b.q_valid].tolist()
    lg = logits[b.q_valid]
    sel, ys, ws = [], [], []
    for i, key in enumerate(zip(tr, st)):
        if key in y:
            sel.append(i); ys.append(float(y[key])); ws.append(w[key])
    if not sel:
        return logits.sum() * 0.0
    dev = logits.device
    lg = lg[torch.tensor(sel, device=dev)]
    loss = F.binary_cross_entropy_with_logits(lg.float(), torch.tensor(ys, device=dev), reduction="none")
    return (loss * torch.tensor(ws, device=dev)).sum()
