#!/usr/bin/env python3
"""Generation with a live step scorer (read_token_v1 Stage J): the read token riding on the
policy's own decode calls, or Qwen2.5-Math-PRM-7B scoring each step beside it.

RowCache holds the key/value columns of B rows, each row filled from column 0 at its own
pace (continuous batching: a finished row is refilled with the next job), with per-column
metadata: the slot kind (PAD, READ, MARKER, TEXT), the text index (text) and the step
ordinal (markers and reads). Engine.forward writes every slot's keys and values to its
column first, then each slot attends to the columns its kind may see:

  TEXT   at text index t: text columns with index <= t (the policy never sees a read)
  MARKER of step k, at the step's first index s: text columns with index < s, itself
  READ   of step k, after its separator (pk text tokens): text columns with index < pk,
         the reads of steps 0..k-1 (chain), the marker of step k, itself
  PAD    itself only

which is the visibility of read_token_inline.text_pass (marker, chain) with the reads kept
in the row's own columns instead of a side cache. Columns are not in position order;
RoPE positions are given per slot.

GraphRunner captures Engine.forward plus an output function as CUDA graphs over one
RowCache, one per (slots, column bucket). Warmup and capture write only the last column,
which no bucket reaches.
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from scripts.inline_fast import MARKER, PAD, READ, TEXT

FIELDS = ("ids", "kind", "pos", "cols", "qlim", "qord")


class RowCache:
    """Key/value columns and column metadata of B rows; n[b] columns of row b are in use."""

    def __init__(self, model, B: int, cap: int, dev):
        cfg = model.config
        H, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
        L, dt = cfg.num_hidden_layers, model.dtype
        self.k = [torch.zeros(B, Hkv, cap, D, dtype=dt, device=dev) for _ in range(L)]
        self.v = [torch.zeros(B, Hkv, cap, D, dtype=dt, device=dev) for _ in range(L)]
        self.kind = torch.zeros(B, cap, dtype=torch.long, device=dev)
        self.tpos = torch.zeros(B, cap, dtype=torch.long, device=dev)
        self.ord = torch.zeros(B, cap, dtype=torch.long, device=dev)
        self.n = [0] * B
        self.B, self.cap = B, cap

    def nbytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in self.k + self.v)

    def reset(self, b: int) -> None:
        self.kind[b].zero_()
        self.n[b] = 0

    def rows(self, lo: int, hi: int) -> "RowCache":
        """Rows lo:hi as views (writes go to this cache)."""
        c = object.__new__(RowCache)
        c.k, c.v = [x[lo:hi] for x in self.k], [x[lo:hi] for x in self.v]
        c.kind, c.tpos, c.ord = self.kind[lo:hi], self.tpos[lo:hi], self.ord[lo:hi]
        c.n, c.B, c.cap = self.n[lo:hi], hi - lo, self.cap
        return c

    def gather(self, rows: list[int], ncol: int, junk: bool = False) -> "RowCache":
        """Copies of the first ncol columns of the given rows, plus one more column for pad
        slots to write to when junk (its index, ncol, is the copy's cap - 1)."""
        idx = torch.tensor(rows, device=self.kind.device)
        w = ncol + 1 if junk else ncol
        c = object.__new__(RowCache)
        c.k = [x[idx, :, :w] for x in self.k]
        c.v = [x[idx, :, :w] for x in self.v]
        c.kind, c.tpos, c.ord = (x[idx, :w] for x in (self.kind, self.tpos, self.ord))
        c.n, c.B, c.cap = [self.n[b] for b in rows], len(rows), w
        return c

    def put_back(self, sub: "RowCache", rows: list[int], ncol: int) -> None:
        idx = torch.tensor(rows, device=self.kind.device)
        for dst, src in zip(self.k + self.v, sub.k + sub.v):
            dst[idx, :, :ncol] = src[:, :, :ncol]
        for dst, src in ((self.kind, sub.kind), (self.tpos, sub.tpos), (self.ord, sub.ord)):
            dst[idx, :ncol] = src[:, :ncol]


def visibility(ck, ctp, cord, kind, qlim, qord, cols) -> torch.Tensor:
    """(B, S, C) bool: which of a row's C columns each of its S slots attends to. ck, ctp,
    cord: column kind, text index, step ordinal (B, C); kind, qlim, qord, cols: per slot
    (B, S), qlim = how many text tokens it sees, cols = its own column."""
    C = ck.shape[1]
    ck, ctp, cord = ck[:, None], ctp[:, None], cord[:, None]
    qk, ql, qo = kind[..., None], qlim[..., None], qord[..., None]
    own = torch.arange(C, device=ck.device) == cols[..., None]
    see = (ck == TEXT) & (ctp < ql)
    see = see | ((qk == READ) & (((ck == READ) & (cord < qo)) | ((ck == MARKER) & (cord == qo))))
    return (see & (qk != PAD)) | own


class Engine:
    """The decoder as a plain layer loop over a RowCache, with the read token's and the start
    marker's soft embeddings and per-layer offsets at their slots (rt, mk: None for none)."""

    def __init__(self, model, rt=None, mk=None):
        self.model, self.inner = model, model.model
        self.layers = list(self.inner.layers)
        cfg = model.config
        self.H, self.Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.G = self.H // self.Hkv
        self.rope = sys.modules[type(self.layers[0].self_attn).__module__].apply_rotary_pos_emb
        dt, L, d = model.dtype, len(self.layers), cfg.hidden_size
        dev = next(model.parameters()).device
        zero = torch.zeros(L, d, dtype=dt, device=dev)
        self.soft = torch.zeros(d, dtype=dt, device=dev) if rt is None else rt.soft.detach().to(dt).reshape(d)
        self.off_read = zero if rt is None or rt.deep is None else rt.deep.detach().to(dt).reshape(L, d)
        self.mk_soft = torch.zeros(d, dtype=dt, device=dev) if mk is None else mk.soft.detach().to(dt).reshape(d)
        self.off_mk = zero if mk is None or mk.deep is None else mk.deep.detach().to(dt).reshape(L, d)
        self.rt, self.dtype, self.dev = rt, dt, dev

    @torch.no_grad()
    def forward(self, c: RowCache, ids, kind, pos, cols, qlim, qord, ncol: int) -> torch.Tensor:
        """Final-norm states of every slot, (B, S, d); attention over the first ncol columns
        (every slot's column must be below ncol)."""
        B, S = kind.shape
        H, Hkv, G, D = self.H, self.Hkv, self.G, self.D
        c.kind.scatter_(1, cols, kind)
        c.tpos.scatter_(1, cols, pos)
        c.ord.scatter_(1, cols, qord)
        see = visibility(c.kind[:, :ncol], c.tpos[:, :ncol], c.ord[:, :ncol], kind, qlim, qord, cols)
        mask = see[:, None].repeat(1, 1, G, 1)  # query row g * S + s is slot s
        is_t = (kind == TEXT).unsqueeze(-1).to(self.dtype)
        is_r = (kind == READ).unsqueeze(-1).to(self.dtype)
        is_m = (kind == MARKER).unsqueeze(-1).to(self.dtype)
        h = is_t * self.inner.embed_tokens(ids) + is_r * self.soft + is_m * self.mk_soft
        cos, sin = self.inner.rotary_emb(h, pos)
        idx = cols[:, None, :, None].expand(B, Hkv, S, D)
        for l, layer in enumerate(self.layers):
            a = layer.self_attn
            y = layer.input_layernorm(h)
            q = a.q_proj(y).view(B, S, H, D)
            k = a.k_proj(y).view(B, S, Hkv, D)
            if hasattr(a, "q_norm"):
                q, k = a.q_norm(q), a.k_norm(k)
            v = a.v_proj(y).view(B, S, Hkv, D).transpose(1, 2)
            q, k = self.rope(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
            c.k[l].scatter_(2, idx, k.contiguous())
            c.v[l].scatter_(2, idx, v.contiguous())
            o = F.scaled_dot_product_attention(q.reshape(B, Hkv, G * S, D), c.k[l][:, :, :ncol],
                                               c.v[l][:, :, :ncol], attn_mask=mask, scale=a.scaling)
            o = o.reshape(B, H, S, D).transpose(1, 2).reshape(B, S, H * D)
            h = h + a.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            h = h + is_r * self.off_read[l] + is_m * self.off_mk[l]
        return self.inner.norm(h)


class GraphRunner:
    """Engine.forward and out_fn(h) captured as CUDA graphs over one RowCache, one per
    (S, ncol); run() copies one packed (6, B, S) input into the graph's buffer and replays.
    The outputs are the graph's own tensors, valid until the next replay of any graph."""

    def __init__(self, eng: Engine, cache: RowCache, out_fn, kernels=None):
        self.eng, self.cache, self.out_fn = eng, cache, out_fn
        self.kernels = kernels
        self.graphs: dict = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_sec = 0.0

    def _graph(self, S: int, ncol: int):
        key = (S, ncol)
        if key in self.graphs:
            return self.graphs[key]
        if ncol > self.cache.cap - 1:
            raise ValueError(f"{ncol} columns reach the junk column of a {self.cache.cap}-column cache")
        t0 = time.perf_counter()
        packed = torch.zeros(len(FIELDS), self.cache.B, S, dtype=torch.long, device=self.eng.dev)
        packed[FIELDS.index("cols")] = self.cache.cap - 1  # warmup and capture write the junk column
        st = dict(zip(FIELDS, packed.unbind(0)))
        ctx = self.kernels if self.kernels is not None else _nullctx
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), ctx():
            for _ in range(2):
                self.out_fn(self.eng.forward(self.cache, ncol=ncol, **st))
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool), ctx():
            out = self.out_fn(self.eng.forward(self.cache, ncol=ncol, **st))
        torch.cuda.synchronize()
        self.capture_sec += time.perf_counter() - t0
        self.graphs[key] = (g, packed, out)
        return self.graphs[key]

    def run(self, packed_host: torch.Tensor, ncol: int):
        g, packed, out = self._graph(packed_host.shape[2], ncol)
        packed.copy_(packed_host, non_blocking=True)
        g.replay()
        return out


def bucket(n: int, step: int) -> int:
    return -(-n // step) * step


def sample(logits: torch.Tensor, u: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Temperature, top-k, then top-p over the top-k softmax (Hugging Face's order), drawn
    by inverting the CDF at u (B,), one uniform per row: (B,) token ids."""
    if temperature <= 0:
        return logits.argmax(-1)
    v, i = (logits.float() / temperature).topk(top_k, dim=-1)
    p = torch.softmax(v, -1)
    p = p * ((p.cumsum(-1) - p) < top_p)  # the smallest prefix reaching top_p
    cdf = p.cumsum(-1)
    j = torch.searchsorted(cdf, (u.to(cdf) * cdf[:, -1]).unsqueeze(-1)).clamp_max(top_k - 1)
    return i.gather(-1, j).squeeze(-1)


class StepState:
    """Blank-line step boundaries on a token stream: a step ends at the token that completes
    a newline, optional spaces, a newline after some non-space text (generate_onpolicy_steps
    splits on \\n\\s*\\n and strips)."""

    def __init__(self):
        self.content, self.nl = False, 0

    def feed(self, s: str) -> bool:
        for ch in s:
            if ch == "\n":
                self.nl += 1
            elif not ch.isspace():
                self.content, self.nl = True, 0
        if self.content and self.nl >= 2:
            self.content, self.nl = False, 0
            return True
        return False


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
