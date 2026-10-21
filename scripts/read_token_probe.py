#!/usr/bin/env python3
"""A learned read token on a frozen policy model (read_token_v0, read_token_v1).

§21.31 found that at the separator token Qwen2.5-Math-PRM-7B's head reads,
correctness is the first principal component and closed-form LDA matches the
head, while in every untrained backbone the same signal is a low-variance margin
read from the step's own tokens (Qwen3-8B best LDA calib-20 0.568). The PRM got
there by full fine-tuning. This asks whether a frozen Qwen3-8B can be given the
same read position by training only m appended soft-token embeddings (m x 4096
parameters) and a linear head on the final state of the last one.

Causality makes this cheap: no earlier position attends to the appended tokens,
so the prefix (verifier template + step) runs once without gradients, and the
backward pass covers only the m appended positions over the cached keys and
values. Batches are left-padded with explicit position ids so every row's soft
tokens sit at the end with their true positions.

Phases: `train` (soft tokens + head, BCE, label 1 = incorrect) writes
`read_token.pt`; `encode` writes the read-token states (final layer) and head
scores for train-subsample/val/test/ProcessBench as in
scripts/encode_pooled_multilayer.py, so scripts/analysis/backbone_geometry.py
scores them unchanged (rep `sep`, layer -1).

read_token_v1 options, all off by default so v0 runs are unchanged:
  --freeze_soft   the embeddings stay at their initialisation; only the head
                  trains (the frozen-slot control, B1)
  --read mean     the head reads the mean of all m final states (B7)
  --head mlp      a one-hidden-layer MLP head (h1024) instead of the linear one
  --deep          a trained vector per decoder layer is added to the hidden state
                  at the read positions only (Stage D; zero-initialised, so it
                  starts exactly at v0)
  --keep_all      encode also stores all m states (`sep_all`) and their mean
                  (`sep_mean`, read by backbone_geometry.py as its own rep)
Plans: docs/read_token_v0_plan.md, docs/read_token_v1_plan.md.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.encode_pooled_multilayer import pb_items, prm800k_items  # noqa: E402
from src.harness.learners import MLPHead  # noqa: E402
from src.onpolicy.prompts import verifier_prefix  # noqa: E402


class ReadToken(torch.nn.Module):
    def __init__(self, m: int, d: int, init: torch.Tensor, read: str = "last",
                 head: str = "linear", n_layers: int = 0):
        super().__init__()
        self.soft = torch.nn.Parameter(init.clone().float())  # (m, d)
        self.head = torch.nn.Linear(d, 1) if head == "linear" else MLPHead(d, hidden=1024)
        self.m, self.read, self.head_kind = m, read, head
        # (n_layers, m, d) residual offsets at the read positions; None for v0.
        self.deep = torch.nn.Parameter(torch.zeros(n_layers, m, d)) if n_layers else None

    def logit(self, h: torch.Tensor) -> torch.Tensor:
        out = self.head(h)
        return out.squeeze(-1) if out.dim() == 2 else out


@contextlib.contextmanager
def deep_hooks(model, rt: ReadToken):
    """Add rt.deep[l] to decoder layer l's output at the last m positions."""
    if rt.deep is None:
        yield
        return
    m = rt.m

    def make(l):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            d = rt.deep[l].to(h.dtype)[None]
            h = torch.cat([h[:, :-m], h[:, -m:] + d], 1)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    handles = [layer.register_forward_hook(make(l)) for l, layer in enumerate(model.model.layers)]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def tokenize(tok, it: dict) -> list[int]:
    pre = tok(verifier_prefix(it["problem"], it["prefix"]), add_special_tokens=True)["input_ids"]
    return pre + tok(it["step"], add_special_tokens=False)["input_ids"]


def batches(lengths: list[int], max_tokens: int, shuffle_seed: int | None):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, i = [], 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * lengths[order[j]] <= max_tokens:
            j += 1
        j = max(j, i + 1)
        out.append(order[i:j])
        i = j
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(out)
    return out


@contextlib.contextmanager
def soft_kernels(soft_attn: str | None):
    """no_cudnn: SDPA without its cuDNN backend (flash, memory-efficient or math)."""
    if soft_attn != "no_cudnn":
        yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        yield


@contextlib.contextmanager
def capture_hooks(model, blocks: list[int], out: dict):
    """Record decoder block l's output at the last position into out[l], (B, d).

    Entered inside deep_hooks, so these hooks run after the offsets are added and
    record the state the next block actually reads (hidden_states[l + 1] in the
    Hugging Face numbering, where hidden_states[0] is the embedding output)."""
    def make(l):
        def hook(_mod, _inp, o):
            h = o[0] if isinstance(o, tuple) else o
            out[l] = h[:, -1].float()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make(l)) for l in blocks]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def read_all_states(model, rt: ReadToken, ids_list: list[list[int]], pad: int, dev, grad: bool,
                    soft_attn: str | None = None, capture: dict | None = None,
                    capture_prefix: dict | None = None):
    """Final-layer (post-norm) states at all m read tokens, (B, m, d), left-padded batch.

    soft_attn="no_cudnn" keeps SDPA for the m-token soft pass but excludes its cuDNN
    backend. On Qwen2.5-Math-7B-Instruct the cuDNN backward returns NaN gradients on
    some left-padded batches (read_token_v1 Stage C, job 629664). Eager attention
    avoids the NaN but drifts from SDPA by 7-8% in bf16 on that model (job 630734
    failed the cache gate), while SDPA without cuDNN stays within 2%.

    capture: {block: None} to also fill in each listed block's output at the last
    read token (pre-norm, offsets included), see capture_hooks.
    capture_prefix: the same at the last token of the text (the last step token,
    which the read token never changes), plus "final": its post-norm state.
    """
    B, L = len(ids_list), max(len(t) for t in ids_list)
    ids = torch.full((B, L), pad, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for r, t in enumerate(ids_list):
        ids[r, L - len(t):] = torch.tensor(t)
        att[r, L - len(t):] = 1
    ids, att = ids.to(dev), att.to(dev)
    pos = (att.cumsum(1) - 1).clamp(min=0)
    pblocks = sorted(k for k in capture_prefix if k != "final") if capture_prefix is not None else []
    with torch.no_grad(), capture_hooks(model, pblocks, capture_prefix if capture_prefix is not None else {}):
        out = model.model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True)
    if capture_prefix is not None and "final" in capture_prefix:
        capture_prefix["final"] = out.last_hidden_state[:, -1].float()
    cache = out.past_key_values
    lens = att.sum(1)
    m = rt.m
    soft_pos = lens[:, None] + torch.arange(m, device=dev)[None]
    att2 = torch.cat([att, torch.ones((B, m), dtype=att.dtype, device=dev)], 1)
    emb = rt.soft.to(model.dtype)[None].expand(B, -1, -1)
    blocks = sorted(capture) if capture is not None else []
    with torch.set_grad_enabled(grad), deep_hooks(model, rt), soft_kernels(soft_attn), \
            capture_hooks(model, blocks, capture if capture is not None else {}):
        out2 = model.model(inputs_embeds=emb, attention_mask=att2, position_ids=soft_pos,
                           past_key_values=cache, use_cache=True)
    return out2.last_hidden_state.float()


def reduce_states(rt: ReadToken, H: torch.Tensor) -> torch.Tensor:
    return H.mean(1) if rt.read == "mean" else H[:, -1]


def read_states(model, rt: ReadToken, ids_list: list[list[int]], pad: int, dev, grad: bool,
                soft_attn: str | None = None):
    """The state the head reads: the last read token, or the mean of all m (--read mean)."""
    return reduce_states(rt, read_all_states(model, rt, ids_list, pad, dev, grad, soft_attn))


def full_forward_states(model, rt: ReadToken, ids: list[int], dev) -> torch.Tensor:
    """Reference for the cache gate: one plain forward over prefix + soft tokens, (m, d)."""
    e = model.get_input_embeddings()(torch.tensor([ids], device=dev))
    full = torch.cat([e, rt.soft.to(model.dtype)[None]], 1)
    with deep_hooks(model, rt):
        return model.model(inputs_embeds=full).last_hidden_state[0, -rt.m:].float()


def make_optimizer(rt: ReadToken, lr: float, freeze_soft: bool):
    groups = [{"params": rt.head.parameters(), "lr": lr, "weight_decay": 0.0}]
    if not freeze_soft:
        groups.insert(0, {"params": [rt.soft], "lr": lr})
    if rt.deep is not None:
        groups.append({"params": [rt.deep], "lr": lr})
    rt.soft.requires_grad_(not freeze_soft)
    return torch.optim.AdamW(groups, weight_decay=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["train", "encode"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--m", type=int, default=4)
    p.add_argument("--init", choices=["vocab", "text"], default="vocab")
    p.add_argument("--init_text", default="\n\nIs this step correct?")
    p.add_argument("--freeze_soft", action="store_true")
    p.add_argument("--read", choices=["last", "mean"], default="last")
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--soft_attn", choices=["no_cudnn"], default=None,
                   help="soft-token pass only: SDPA without the cuDNN backend (default: any SDPA backend)")
    p.add_argument("--keep_all", action="store_true", help="encode: also store all m states")
    p.add_argument("--train", type=Path)
    p.add_argument("--n_train", type=int, default=40000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--max_tokens", type=int, default=16000)
    p.add_argument("--max_seq_len", type=int, default=3072)
    p.add_argument("--prm800k", nargs="*", default=[], help="encode: name:path[:limit]")
    p.add_argument("--pb", nargs="*", default=[], help="encode: subset:path")
    p.add_argument("--ckpt", type=Path, help="encode: read_token.pt (default: OUT/read_token.pt)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    emb = model.get_input_embeddings().weight

    if a.phase == "train":
        if a.init == "text":
            tids = tok(a.init_text, add_special_tokens=False)["input_ids"]
            tids = (tids * a.m)[:a.m] if len(tids) < a.m else tids[-a.m:]
            init = emb[torch.tensor(tids, device=emb.device)].detach().float()
        else:
            g = torch.Generator().manual_seed(a.seed)
            idx = torch.randint(0, emb.shape[0], (a.m,), generator=g)
            init = emb[idx.to(emb.device)].detach().float()
        rt = ReadToken(a.m, d, init.cpu(), read=a.read, head=a.head,
                       n_layers=n_layers if a.deep else 0).to(dev)
        print(f"[config] model={a.model} m={a.m} init={a.init} read={a.read} head={a.head} "
              f"deep={a.deep} freeze_soft={a.freeze_soft} n_train={a.n_train} seed={a.seed}", flush=True)
        items = prm800k_items(a.train, a.n_train, a.seed)
        if a.smoke:
            items = items[:a.smoke]
        toks = [tokenize(tok, it) for it in items]
        keep = [i for i, t in enumerate(toks) if len(t) <= a.max_seq_len]
        toks = [toks[i] for i in keep]
        # Gate: the cached left-padded read states must equal a plain full forward.
        with torch.no_grad():
            got = read_all_states(model, rt, toks[:2], pad, dev, grad=False, soft_attn=a.soft_attn)
            for r in range(2):
                ref = full_forward_states(model, rt, toks[r], dev)
                rel = float((got[r] - ref).norm() / ref.norm())
                print(f"[gate] row {r}: relative difference {rel:.4f}", flush=True)
                if not rel <= 0.05:  # also catches NaN
                    raise SystemExit("cached read state does not match the full forward")
        y = torch.tensor([items[i]["label"] for i in keep], dtype=torch.float32)
        opt = make_optimizer(rt, a.lr, a.freeze_soft)
        params = [q for q in rt.parameters() if q.requires_grad]
        # With frozen embeddings and no deep offsets the read state does not depend
        # on anything trained, so the backbone runs without gradients.
        state_grad = not a.freeze_soft or rt.deep is not None
        bl = batches([len(t) for t in toks], a.max_tokens, a.seed)
        steps = a.epochs * len(bl)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
        t0, step, run = time.perf_counter(), 0, 0.0
        for ep in range(a.epochs):
            for bi, b in enumerate(bl if ep == 0 else batches([len(t) for t in toks], a.max_tokens, a.seed + ep)):
                h = read_states(model, rt, [toks[i] for i in b], pad, dev, grad=state_grad,
                                soft_attn=a.soft_attn)
                logit = rt.logit(h)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y[b].to(dev))
                if not torch.isfinite(loss):
                    raise SystemExit(f"non-finite loss at step {step}")
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); step += 1
                run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
                if step % 50 == 0 or step == steps:
                    print(f"[train m={a.m}] step {step}/{steps} loss {run:.4f} "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)
        torch.save({"soft": rt.soft.detach().cpu(), "head": rt.head.state_dict(), "m": a.m,
                    "model": a.model, "init": a.init, "read": a.read, "head_kind": a.head,
                    "freeze_soft": a.freeze_soft, "seed": a.seed, "n_train": a.n_train,
                    "soft_attn": a.soft_attn,
                    "deep": None if rt.deep is None else rt.deep.detach().cpu()},
                   a.out / "read_token.pt")
        return

    ck = torch.load(a.ckpt or a.out / "read_token.pt", weights_only=False)
    deep = ck.get("deep")
    rt = ReadToken(ck["m"], d, ck["soft"], read=ck.get("read", "last"), head=ck.get("head_kind", "linear"),
                   n_layers=0 if deep is None else deep.shape[0]).to(dev)
    rt.head.load_state_dict(ck["head"])
    if deep is not None:
        rt.deep.data.copy_(deep.to(dev))
    rt.eval()
    soft_attn = a.soft_attn or ck.get("soft_attn")
    splits = []
    for spec in a.prm800k:
        name, path, *lim = spec.split(":")
        splits.append((name, prm800k_items(Path(path), int(lim[0]) if lim else 0, a.seed + 1)))
    for spec in a.pb:
        sub, path = spec.split(":")
        splits.append((f"pb_{sub}", pb_items(Path(path), sub)))
    for name, items in splits:
        if a.smoke:
            items = items[:a.smoke]
        toks = [tokenize(tok, it) for it in items]
        keep = [i for i, t in enumerate(toks) if len(t) <= a.max_seq_len]
        H = np.zeros((len(items), 1, d), np.float16)
        if a.keep_all:
            HA = np.zeros((len(items), rt.m, d), np.float16)
            HM = np.zeros((len(items), 1, d), np.float16)
        S = np.full(len(items), np.nan, np.float32)
        t0 = time.perf_counter()
        for b in batches([len(toks[i]) for i in keep], a.max_tokens, None):
            rows = [keep[i] for i in b]
            with torch.no_grad():
                hall = read_all_states(model, rt, [toks[i] for i in rows], pad, dev, grad=False,
                                       soft_attn=soft_attn)
                h = reduce_states(rt, hall)
                s = torch.sigmoid(rt.logit(h)).cpu().numpy()
            H[rows, 0] = h.cpu().numpy()
            if a.keep_all:
                HA[rows] = hall.cpu().numpy()
                HM[rows, 0] = hall.mean(1).cpu().numpy()
            S[rows] = s
        extra = {"sep_all": HA, "sep_mean": HM} if a.keep_all else {}
        np.savez(a.out / f"{name}.shard0.npz", layers=np.array([-1]),
                 y=np.array([it["label"] for it in items], np.int8),
                 kept=np.isin(np.arange(len(items)), keep), head_p_correct=1 - S, sep=H, **extra)
        with open(a.out / f"{name}.shard0.meta.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps({k: it[k] for k in ("uid", "id", "step_idx", "first_error", "label")
                                    if k in it}) + "\n")
        print(f"[encode] {name}: {len(keep)}/{len(items)} in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
