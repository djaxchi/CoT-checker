#!/usr/bin/env python3
"""Fast inline read: read_token_inline.read_pass as a plain loop over the decoder layers,
captured as CUDA graphs.

read_pass runs the read call through the Hugging Face forward: mask processing, cache
update, a Python hook per layer for the offsets. With 35 to 70 read tokens that call is
launch-bound, not compute-bound. FastReader computes the same states from the text
cache's key/value tensors directly:

  * each layer: RMSNorm, q/k/v projections (q_norm and k_norm on Qwen3), RoPE, one SDPA
    over [text keys; slot keys] under the read layout's mask, o_proj, MLP;
  * grouped-query attention without repeating the keys: the G query heads sharing a
    key/value head are folded into the query length, the mask repeated G times;
  * the offsets added in place of the hooks, from each slot's kind (read token, start
    marker, or a text token for the decode-step comparison);
  * the text cache is only read, never appended to.

GraphedReader captures FastReader per (rows, text length bucket, slot bucket) into CUDA
graphs sharing one memory pool, copies the text cache into the bucket's static buffers
and replays. Padded text columns are masked out; padded slots see only themselves.

Slot kinds: 0 padding, 1 read token, 2 start marker, 3 text token (input id given).
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

PAD, READ, MARKER, TEXT = 0, 1, 2, 3


def layout_kinds(lay: dict) -> torch.Tensor:
    """The slot kinds of a text_pass layout, (B, S): markers then read tokens with a marker,
    read tokens only without (padded read slots stay read slots, as in read_pass)."""
    B, K = lay["pos"].shape[0], lay["K"]
    kind = torch.full((B, K), READ, dtype=torch.long, device=lay["pos"].device)
    if lay["marker"]:
        kind = torch.cat([torch.full_like(kind, MARKER), kind], 1)
    return kind


def cache_kv(cache) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return [c.keys for c in cache.layers], [c.values for c in cache.layers]


class FastReader:
    """The read call of read_pass over a text cache, without the Hugging Face forward."""

    def __init__(self, model, rt, mk=None):
        if rt.m != 1:
            raise ValueError("inline reads need an m = 1 read token")
        self.inner = model.model
        self.layers = list(self.inner.layers)
        cfg = model.config
        self.H, self.Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.G = self.H // self.Hkv
        self.rope = sys.modules[type(self.layers[0].self_attn).__module__].apply_rotary_pos_emb
        dt, L, d = model.dtype, len(self.layers), cfg.hidden_size
        dev = next(model.parameters()).device
        zero = torch.zeros(L, d, dtype=dt, device=dev)
        self.soft = rt.soft.detach().to(dt).reshape(d)
        self.off_read = zero if rt.deep is None else rt.deep.detach().to(dt).reshape(L, d)
        self.mk_soft = torch.zeros(d, dtype=dt, device=dev) if mk is None else mk.soft.detach().to(dt).reshape(d)
        self.off_mk = zero if mk is None or mk.deep is None else mk.deep.detach().to(dt).reshape(L, d)
        self.dtype = dt

    @torch.no_grad()
    def forward(self, ks: list[torch.Tensor], vs: list[torch.Tensor], kind: torch.Tensor,
                pos: torch.Tensor, mask: torch.Tensor, ids: torch.Tensor | None = None,
                inplace: bool = False) -> torch.Tensor:
        """Final-norm states of every slot, (B, S, d).

        ks, vs: per layer (B, Hkv, T, D); kind, pos: (B, S); mask: (B, 1, S, T + S) bool;
        ids: (B, S) input ids for text slots (kind 3). inplace: ks, vs hold T + S columns
        and the slots' keys and values are written into the last S (no concatenation)."""
        B, S = kind.shape
        H, Hkv, G, D = self.H, self.Hkv, self.G, self.D
        is_r = (kind == READ).unsqueeze(-1).to(self.dtype)
        is_m = (kind == MARKER).unsqueeze(-1).to(self.dtype)
        h = is_r * self.soft + is_m * self.mk_soft
        if ids is not None:
            h = h + (kind == TEXT).unsqueeze(-1).to(self.dtype) * self.inner.embed_tokens(ids)
        cos, sin = self.inner.rotary_emb(h, pos)
        mq = mask.repeat(1, 1, G, 1)  # query row g * S + s is slot s
        for l, layer in enumerate(self.layers):
            a = layer.self_attn
            y = layer.input_layernorm(h)
            q = a.q_proj(y).view(B, S, H, D)
            k = a.k_proj(y).view(B, S, Hkv, D)
            if hasattr(a, "q_norm"):
                q, k = a.q_norm(q), a.k_norm(k)
            v = a.v_proj(y).view(B, S, Hkv, D).transpose(1, 2)
            q, k = self.rope(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
            if inplace:
                kl, vl = ks[l], vs[l]
                kl[:, :, -S:] = k
                vl[:, :, -S:] = v
            else:
                kl, vl = torch.cat([ks[l], k], 2), torch.cat([vs[l], v], 2)
            o = F.scaled_dot_product_attention(q.reshape(B, Hkv, G * S, D), kl, vl, attn_mask=mq,
                                               scale=a.scaling)
            o = o.reshape(B, H, S, D).transpose(1, 2).reshape(B, S, H * D)
            h = h + a.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            h = h + is_r * self.off_read[l] + is_m * self.off_mk[l]
        return self.inner.norm(h)

    def read(self, cache, lay: dict) -> torch.Tensor:
        """read_pass's output, (sum of K, d), float."""
        ks, vs = cache_kv(cache)
        h = self.forward(ks, vs, layout_kinds(lay), lay["pos"], lay["mask"])
        return h.float()[lay["rows"], lay["cols"]]


def fuse_views(linears: list) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One weight (and bias) holding several nn.Linear's stacked along the output dim; each
    module's own weight becomes a view into it, so the model is unchanged and no memory is
    duplicated."""
    w = torch.cat([m.weight.data for m in linears], 0)
    b = torch.cat([m.bias.data for m in linears], 0) if linears[0].bias is not None else None
    i = 0
    for m in linears:
        n = m.weight.shape[0]
        m.weight.data = w[i:i + n]
        if b is not None:
            m.bias.data = b[i:i + n]
        i += n
    return w, b


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen2/Qwen3 RMSNorm: in float32, rounded to the input dtype, then the weight."""
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _rot(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin


def _pre(h, ln1, wqkv, bqkv, qn, kn, cos, sin, H: int, Hkv: int, D: int, eps: float):
    """Norm, fused q/k/v projection, q/k head norms, RoPE: q folded as (B, Hkv, G * S, D),
    k and v as (B, Hkv, S, D)."""
    B, S, _ = h.shape
    qkv = F.linear(_rms(h, ln1, eps), wqkv, bqkv)
    q, k, v = qkv.split([H * D, Hkv * D, Hkv * D], -1)
    q, k = q.reshape(B, S, H, D), k.reshape(B, S, Hkv, D)
    if qn is not None:
        q, k = _rms(q, qn, eps), _rms(k, kn, eps)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    q = _rot(q.transpose(1, 2), c, s).reshape(B, Hkv, (H // Hkv) * S, D)
    k = _rot(k.transpose(1, 2), c, s)
    return q, k, v.reshape(B, S, Hkv, D).transpose(1, 2)


def _post(h, o, wo, bo, ln2, wgu, wd, is_r, is_m, off_r, off_m, eps: float):
    """o_proj and residual, norm, fused gate/up, SiLU, down and residual, the offsets."""
    B, S, _ = h.shape
    h = h + F.linear(o, wo, bo)
    g, u = F.linear(_rms(h, ln2, eps), wgu).chunk(2, -1)
    h = h + F.linear(F.silu(g) * u, wd)
    return h + is_r * off_r + is_m * off_m


class FusedReader(FastReader):
    """FastReader with q/k/v and gate/up each one matmul (weights fused in place, no copy)
    and the pointwise work between matmuls compiled into fused kernels (torch.compile);
    attention stays one SDPA call per layer."""

    def __init__(self, model, rt, mk=None, compile: bool = True):
        super().__init__(model, rt, mk)
        if getattr(model.config, "hidden_act", "silu") != "silu":
            raise ValueError("FusedReader assumes a SiLU MLP")
        self.eps = model.config.rms_norm_eps
        self.w = []
        for layer in self.layers:
            a, m = layer.self_attn, layer.mlp
            wqkv, bqkv = fuse_views([a.q_proj, a.k_proj, a.v_proj])
            wgu, _ = fuse_views([m.gate_proj, m.up_proj])
            self.w.append((layer.input_layernorm.weight, wqkv, bqkv,
                           a.q_norm.weight if hasattr(a, "q_norm") else None,
                           a.k_norm.weight if hasattr(a, "k_norm") else None,
                           a.o_proj.weight, a.o_proj.bias, layer.post_attention_layernorm.weight,
                           wgu, m.down_proj.weight, a.scaling))
        if compile:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self.pre, self.post = torch.compile(_pre, dynamic=True), torch.compile(_post, dynamic=True)
        else:
            self.pre, self.post = _pre, _post

    @torch.no_grad()
    def forward(self, ks, vs, kind, pos, mask, ids=None, inplace: bool = False) -> torch.Tensor:
        B, S = kind.shape
        H, Hkv, G, D = self.H, self.Hkv, self.G, self.D
        is_r = (kind == READ).unsqueeze(-1).to(self.dtype)
        is_m = (kind == MARKER).unsqueeze(-1).to(self.dtype)
        h = is_r * self.soft + is_m * self.mk_soft
        if ids is not None:
            h = h + (kind == TEXT).unsqueeze(-1).to(self.dtype) * self.inner.embed_tokens(ids)
        cos, sin = self.inner.rotary_emb(h, pos)
        mq = mask.repeat(1, 1, G, 1)
        for l, (ln1, wqkv, bqkv, qn, kn, wo, bo, ln2, wgu, wd, scale) in enumerate(self.w):
            q, k, v = self.pre(h, ln1, wqkv, bqkv, qn, kn, cos, sin, H, Hkv, D, self.eps)
            if inplace:
                kl, vl = ks[l], vs[l]
                kl[:, :, -S:] = k
                vl[:, :, -S:] = v
            else:
                kl, vl = torch.cat([ks[l], k], 2), torch.cat([vs[l], v], 2)
            o = F.scaled_dot_product_attention(q, kl, vl, attn_mask=mq, scale=scale)
            o = o.reshape(B, H, S, D).transpose(1, 2).reshape(B, S, H * D)
            h = self.post(h, o, wo, bo, ln2, wgu, wd, is_r, is_m, self.off_read[l], self.off_mk[l], self.eps)
        return self.inner.norm(h)


class GraphedReader:
    """FastReader captured as CUDA graphs, one per (B, text bucket, slot bucket)."""

    def __init__(self, fr: FastReader, t_step: int = 256, s_step: int = 16, kernels=None):
        self.fr, self.t_step, self.s_step = fr, t_step, s_step
        self.kernels = kernels  # a context manager factory for the SDPA backends, or None
        self.graphs: dict = {}
        self.kv: dict = {}  # B -> (keys, values) per layer, (B, Hkv, capacity, D)
        self.cap: dict = {}  # B -> text capacity reserved before the first capture
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_sec = 0.0

    def bucket(self, T: int, S: int) -> tuple[int, int]:
        return math.ceil(T / self.t_step) * self.t_step, math.ceil(S / self.s_step) * self.s_step

    def reserve(self, B: int, T: int, S: int) -> None:
        """Size the shared key/value buffers of B-row calls for texts up to T tokens and up
        to S slots (before any capture for B: every graph of B uses a slice of them)."""
        if B in self.kv:
            raise ValueError(f"buffers for B = {B} already exist")
        Tb, Sb = self.bucket(T, S)
        self.cap[B] = max(self.cap.get(B, 0), Tb + Sb)

    def _buffers(self, B: int, n: int):
        if B not in self.kv:
            fr, cap = self.fr, max(n, self.cap.get(B, 0))
            dev = fr.soft.device
            self.kv[B] = tuple([torch.zeros(B, fr.Hkv, cap, fr.D, dtype=fr.dtype, device=dev)
                                for _ in fr.layers] for _ in range(2))
        ks, vs = self.kv[B]
        if ks[0].shape[2] < n:
            raise ValueError(f"{n} key columns exceed the {ks[0].shape[2]} reserved for B = {B}")
        return [x[:, :, :n] for x in ks], [x[:, :, :n] for x in vs]

    def _graph(self, B: int, Tb: int, Sb: int, with_ids: bool):
        key = (B, Tb, Sb, with_ids)
        if key in self.graphs:
            return self.graphs[key]
        import time
        t0 = time.perf_counter()
        fr = self.fr
        st = static_inputs(fr, B, Tb, Sb, with_ids, self._buffers(B, Tb + Sb))
        ctx = self.kernels if self.kernels is not None else _nullctx
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), ctx():
            for _ in range(2):
                fr.forward(**st)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool), ctx():
            out = fr.forward(**st)
        torch.cuda.synchronize()
        self.capture_sec += time.perf_counter() - t0
        self.graphs[key] = (g, st, out)
        return self.graphs[key]

    def run(self, ks, vs, kind, pos, mask, ids=None) -> torch.Tensor:
        """FastReader.forward on padded static buffers, (B, S, d) (a view of the graph's
        output, valid until the next replay of that graph)."""
        B, _, T, _ = ks[0].shape
        S = kind.shape[1]
        Tb, Sb = self.bucket(T, S)
        g, st, out = self._graph(B, Tb, Sb, ids is not None)
        fill_inputs(st, ks, vs, kind, pos, mask, ids)
        g.replay()
        return out[:, :S]

    def read(self, cache, lay: dict) -> torch.Tensor:
        ks, vs = cache_kv(cache)
        h = self.run(ks, vs, layout_kinds(lay), lay["pos"], lay["mask"])
        return h.float()[lay["rows"], lay["cols"]]


def static_inputs(fr: FastReader, B: int, Tb: int, Sb: int, with_ids: bool, kv=None) -> dict:
    """Zeroed FastReader.forward inputs for B rows, Tb text columns and Sb slots; every
    slot sees itself, so padded slots stay finite. The key/value buffers hold Tb + Sb
    columns, the slots written into the last Sb in place; kv: given buffers to use."""
    dev, L = fr.soft.device, len(fr.layers)
    if kv is None:
        kv = tuple([torch.zeros(B, fr.Hkv, Tb + Sb, fr.D, dtype=fr.dtype, device=dev) for _ in range(L)]
                   for _ in range(2))
    st = {"ks": kv[0], "vs": kv[1], "inplace": True,
          "kind": torch.zeros(B, Sb, dtype=torch.long, device=dev),
          "pos": torch.zeros(B, Sb, dtype=torch.long, device=dev),
          "mask": torch.zeros(B, 1, Sb, Tb + Sb, dtype=torch.bool, device=dev),
          "ids": torch.zeros(B, Sb, dtype=torch.long, device=dev) if with_ids else None}
    st["mask"][:, 0, torch.arange(Sb), Tb + torch.arange(Sb)] = True
    return st


def fill_inputs(st: dict, ks, vs, kind, pos, mask, ids=None) -> None:
    """Copy one call's inputs (T text columns, S slots) into static_inputs buffers in place:
    the text columns past T and the slots past S are masked out of every real slot."""
    T, S = ks[0].shape[2], kind.shape[1]
    Tb = st["mask"].shape[-1] - st["kind"].shape[1]
    torch._foreach_copy_([x[:, :, :T] for x in st["ks"]] + [x[:, :, :T] for x in st["vs"]], ks + vs)
    st["kind"].zero_()
    st["kind"][:, :S] = kind
    st["pos"].zero_()
    st["pos"][:, :S] = pos
    m = st["mask"]
    m[:, :, :S].zero_()
    m[:, :, :S, :T] = mask[:, :, :, :T]
    m[:, :, :S, Tb:Tb + S] = mask[:, :, :, T:]
    if ids is not None:
        st["ids"].zero_()
        st["ids"][:, :S] = ids


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
