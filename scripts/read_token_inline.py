#!/usr/bin/env python3
"""The RTH read token inside the trace (read_token_v1 Stage H, Plan B).

RTH (read_token_probe.py) reads a step under the verifier template, rebuilt for
every step ("Problem / Previous reasoning / Current step"), so scoring a
trajectory re-reads it once per step. Here a trajectory is read once, in the
policy's own chat layout, with one read token after each step's separator:

    chat_prompt(problem) s0 \\n\\n [v] s1 \\n\\n [v] ... sK \\n\\n [v]

No text token attends to a read token, so the text states are the ones the
policy computed while writing. Read token k attends to the text up to the end
of step k's separator and to itself, at the position the next text token takes.
That is a read token appended to the generation cache after step k and dropped
after the read, which `read_all_states` computes on the standalone prefix; the
tests and the training gate check the two agree.

Two calls per batch: the text once, without gradients (no text state depends on
anything trained), then every read token of every row in one call over that
cache, with a 4D mask giving each read token its own prefix. The per-layer
offsets apply at every read position. Only m = 1 tokens (RTH) are supported.

--start_marker (Stage H-S) adds a soft token at the start of every step that only
that step's read token attends to, in place of the template's "Current step:"
header; --start_deep gives it per-layer offsets of its own. --chain (Stage H-C) puts a read
token after every step and lets each see the earlier ones (kept in a side cache
online, still invisible to the text), so the previous read token marks where the
step begins.

Items are grouped into sequences: an item rides on a longer item of the same
problem when its text (earlier steps + step) is that item's text up to a
separator, so it reads the same tokens it would read alone. If the tokenizer has
no token boundary at an item's separator inside the longer text, the item is
read as its own sequence instead.

Phases mirror read_token_probe.py: `train` (same items, labels, loss, optimizer
and schedule as RTH; batches capped at RTH's items per optimizer step) writes
read_token.pt; `encode` writes the same npz/meta files as read_token_probe.py
encode, plus `passive_block<l>` and `passive_final`, the text states at the
separator token before each read token (block outputs at --passive_blocks, and
the post-norm final state), for the
passive control that reads the same text without a read token.

    python scripts/read_token_inline.py train --init_text " correct" --deep \\
        --train <data>/prm800k_probe_train_full.jsonl --n_train 100000 --seed 0 --out <dir>
    python scripts/read_token_inline.py encode --out <dir> --prm800k val:<val.jsonl> --pb math:<pb.jsonl>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.encode_pooled_multilayer import pb_items, prm800k_items  # noqa: E402
from scripts.read_token_probe import (  # noqa: E402
    ReadToken, batches, make_optimizer, read_all_states, read_hooks, read_token_from_ckpt,
    soft_kernels, tokenize)
from src.onpolicy.prompts import chat_prompt  # noqa: E402

SEP = "\n\n"
LAYOUTS = {"chat": chat_prompt}


def item_text(it: dict) -> str:
    """The item's earlier steps and its step, as they sit in the trajectory."""
    return (it["prefix"] + SEP if it["prefix"] else "") + it["step"]


def group_items(items: list[dict]) -> list[dict]:
    """Sequences {"problem", "text", "members"}: every item in exactly one, riding on
    the longest item of its problem whose text equals it or continues it after SEP."""
    by_problem = defaultdict(list)
    for i, it in enumerate(items):
        by_problem[it["problem"]].append(i)
    out = []
    for problem, idx in by_problem.items():
        seqs: list[dict] = []
        for i in sorted(idx, key=lambda i: (-len(item_text(items[i])), i)):
            s = item_text(items[i])
            for q in seqs:
                if q["text"] == s or q["text"].startswith(s + SEP):
                    q["members"].append(i)
                    break
            else:
                seqs.append({"problem": problem, "text": s, "members": [i]})
        out += seqs
    return out


class StartMarker(torch.nn.Module):
    """A soft token at the start of every step, seen only by that step's read token (H-S).

    It sits at the step's first position and attends to the text before the step;
    no text token attends to it, so the text states stay the generation states. It
    stands in for the verifier template's "Current step:" header, which tells RTH
    where the step it judges begins. deep: optional per-layer offsets of its own.
    """

    def __init__(self, d: int, init: torch.Tensor, n_layers: int = 0):
        super().__init__()
        self.soft = torch.nn.Parameter(init.clone().float())  # (1, d)
        self.deep = torch.nn.Parameter(torch.zeros(n_layers, 1, d)) if n_layers else None


def marker_from_ckpt(ck: dict, dev) -> StartMarker | None:
    if ck.get("start_soft") is None:
        return None
    deep = ck.get("start_deep")
    mk = StartMarker(ck["start_soft"].shape[1], ck["start_soft"], 0 if deep is None else deep.shape[0])
    if deep is not None:
        with torch.no_grad():
            mk.deep.copy_(deep)
    return mk.to(dev).eval()


@contextlib.contextmanager
def marker_hooks(model, mk: StartMarker | None, lo: int, hi: int):
    """Add mk.deep[l] to decoder layer l's output at positions lo:hi of every call."""
    if mk is None or mk.deep is None:
        yield
        return

    def make(l):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            a, b, _ = slice(lo, hi).indices(h.shape[1])
            h = torch.cat([h[:, :a], h[:, a:b] + mk.deep[l].to(h.dtype)[None], h[:, b:]], 1)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    handles = [layer.register_forward_hook(make(l)) for l, layer in enumerate(model.model.layers)]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def encode_text(tok, text: str) -> tuple[list[int], dict[int, int]]:
    """Token ids and {character end: number of tokens up to the last token ending there}."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ends = {}
    for j, (_, e) in enumerate(enc["offset_mapping"]):
        ends[e] = j + 1
    return list(enc["input_ids"]), ends


def boundary_at_or_before(ends: dict[int, int], c: int) -> int:
    """Tokens up to the last token boundary at or before character c."""
    if c in ends:
        return ends[c]
    before = [e for e in ends if e <= c]
    return ends[max(before)] if before else 0


def chain_slots(prompt: str, text: str, ends: dict[int, int], member_pos: list[int]) -> tuple[list[int], list[int]]:
    """A read slot at every step separator of `text` that ends on a token boundary, and at
    every member position: (slot positions, slot step starts), ascending. A slot's step
    starts where the previous slot sits (the first at the end of the prompt)."""
    seps = {ends[len(prompt) + m.end()] for m in re.finditer(re.escape(SEP), text + SEP)
            if len(prompt) + m.end() in ends}
    slots = sorted(seps | set(member_pos))
    first = boundary_at_or_before(ends, len(prompt))
    return slots, [first] + slots[:-1]


def build_sequences(tok, items: list[dict], layout: str = "chat", chain: bool = False) -> tuple[list[dict], int]:
    """Tokenised sequences {"ids", "pos", "spos", "members", "slot"}: read slot k sees pos[k]
    text tokens and its step starts at token spos[k] (the last token boundary at or before
    the step's first character); member m is read at slot[m]. Without chain every member
    has its own slot. With chain (Stage H-C) there is a slot after every step of the
    sequence, labelled or not, since a read token there also sees the earlier read tokens.
    Also returns how many items had to be read on their own (no token boundary at their
    separator)."""
    prompt_of = LAYOUTS[layout]
    out, alone = [], 0

    def finish(prompt, text, ids, ends, pos, spos, members):
        order = np.argsort(pos, kind="stable")
        pos, spos, members = [pos[o] for o in order], [spos[o] for o in order], [members[o] for o in order]
        if not chain:
            return {"ids": ids, "pos": pos, "spos": spos, "members": members, "slot": list(range(len(pos)))}
        slots, starts = chain_slots(prompt, text, ends, pos)
        return {"ids": ids, "pos": slots, "spos": starts, "members": members,
                "slot": [slots.index(q) for q in pos]}

    for g in group_items(items):
        prompt = prompt_of(g["problem"])
        ids, ends = encode_text(tok, prompt + g["text"] + SEP)
        pos, spos, members = [], [], []
        for i in g["members"]:
            it = items[i]
            start = len(prompt) + (len(it["prefix"]) + len(SEP) if it["prefix"] else 0)
            q = ends.get(len(prompt) + len(item_text(it)) + len(SEP))
            if q is None:
                alone += 1
                own, own_ends = encode_text(tok, prompt + item_text(it) + SEP)
                out.append(finish(prompt, item_text(it), own, own_ends, [len(own)],
                                  [boundary_at_or_before(own_ends, start)], [i]))
                continue
            pos.append(q)
            spos.append(boundary_at_or_before(ends, start))
            members.append(i)
        if members:
            out.append(finish(prompt, g["text"], ids, ends, pos, spos, members))
    return out, alone


def member_index(seqs: list[dict], dev=None) -> torch.Tensor:
    """For a batch of built sequences, the row of each member in inline_read's output
    (one row per slot, row by row), in member order."""
    idx, off = [], 0
    for q in seqs:
        idx += [off + k for k in q["slot"]]
        off += len(q["pos"])
    return torch.tensor(idx, dtype=torch.long, device=dev)


def gather_hooks(model, blocks: list[int], rows: torch.Tensor, cols: torch.Tensor, out: dict):
    """Record decoder block l's output at positions (rows, cols) into out[l], (n, d)."""
    def make(l):
        def hook(_mod, _inp, o):
            h = o[0] if isinstance(o, tuple) else o
            out[l] = h[rows, cols].float()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make(l)) for l in blocks]
    return handles


def text_pass(model, seqs: list[tuple], pad: int, dev, passive: dict | None = None,
              marker: bool = False, chain: bool = False) -> tuple[object, dict]:
    """The text of every row, once, without gradients: (its cache, the read layout).

    seqs: (text ids, pos) or (text ids, pos, spos) per row; read token k sees pos[k]
    text tokens. With marker, the read call holds K start markers then K read tokens:
    marker k at position spos[k] sees the text before it, read token k sees its text,
    its own marker and itself. With chain, read token k also sees read tokens 0..k-1 of
    its row (they stay in a side cache instead of being dropped).
    passive: {block: None, "final": None} to also fill in the text states at token
    pos[k] - 1 (the separator the read token follows), block outputs and post-norm.
    """
    B = len(seqs)
    T = max(len(r[0]) for r in seqs)
    K = max(len(r[1]) for r in seqs)
    S = 2 * K if marker else K
    ids = torch.full((B, T), pad, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    mask = torch.zeros((B, 1, S, T + S), dtype=torch.bool)
    pos2 = torch.zeros((B, S), dtype=torch.long)
    rows, cols, tpos = [], [], []
    for b, r in enumerate(seqs):
        t, p = r[0], r[1]
        if not p or max(p) > len(t) or min(p) < 1:
            raise ValueError(f"row {b}: read positions {p} outside 1..{len(t)}")
        if marker and (len(r) < 3 or any(not 0 <= sp < pk for sp, pk in zip(r[2], p))):
            raise ValueError(f"row {b}: start markers need 0 <= spos < pos")
        ids[b, :len(t)] = torch.tensor(t)
        att[b, :len(t)] = 1
        off = K if marker else 0
        for k, pk in enumerate(p):
            mask[b, 0, off + k, :pk] = True
            pos2[b, off + k] = pk
            if chain:
                mask[b, 0, off + k, T + off:T + off + k] = True  # the earlier read tokens
            if marker:
                sp = r[2][k]
                mask[b, 0, k, :sp] = True
                pos2[b, k] = sp
                mask[b, 0, off + k, T + k] = True  # the read token sees its step's marker
            rows.append(b)
            cols.append(off + k)
            tpos.append(pk - 1)
        mask[b, 0, torch.arange(S), T + torch.arange(S)] = True  # every slot sees itself
    rows, cols = torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)
    tcols = torch.tensor(tpos, device=dev)
    blocks = sorted(k for k in passive if k != "final") if passive is not None else []
    handles = gather_hooks(model, blocks, rows, tcols, passive) if blocks else []
    try:
        with torch.no_grad():
            out = model.model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=True)
    finally:
        for hd in handles:
            hd.remove()
    if passive is not None and "final" in passive:
        passive["final"] = out.last_hidden_state[rows, tcols].float()
    return out.past_key_values, {"mask": mask.to(dev), "pos": pos2.to(dev), "rows": rows, "cols": cols,
                                 "K": K, "marker": marker, "chain": chain}


def read_pass(model, rt: ReadToken, cache, lay: dict, grad: bool, soft_attn: str | None = None,
              mk: StartMarker | None = None) -> torch.Tensor:
    """Every read token (and start marker) over a text_pass cache in one call, (sum of K, d).
    Appends to the cache."""
    if rt.m != 1:
        raise ValueError("inline reads need an m = 1 read token")
    if lay["marker"] != (mk is not None):
        raise ValueError("the layout and the start marker disagree")
    B, K = lay["pos"].shape[0], lay["K"]
    emb = rt.soft.to(model.dtype)[None].expand(B, K, -1)
    if mk is not None:
        emb = torch.cat([mk.soft.to(model.dtype)[None].expand(B, K, -1), emb], 1)
    with torch.set_grad_enabled(grad), read_hooks(model, rt, n=K), marker_hooks(model, mk, 0, K), \
            soft_kernels(soft_attn):
        h = model.model(inputs_embeds=emb, attention_mask=lay["mask"], position_ids=lay["pos"],
                        past_key_values=cache, use_cache=True).last_hidden_state
    return h.float()[lay["rows"], lay["cols"]]


def inline_read(model, rt: ReadToken, seqs: list[tuple], pad: int, dev, grad: bool,
                soft_attn: str | None = None, passive: dict | None = None,
                mk: StartMarker | None = None, chain: bool = False) -> torch.Tensor:
    """Final-layer (post-norm) states of every read token, (sum of K, d), row by row
    (text_pass, then read_pass)."""
    if rt.m != 1:
        raise ValueError("inline reads need an m = 1 read token")
    cache, lay = text_pass(model, seqs, pad, dev, passive, marker=mk is not None, chain=chain)
    return read_pass(model, rt, cache, lay, grad, soft_attn, mk)


def uncached_reference(model, rt: ReadToken, ids: list[int], pos: list[int], dev,
                       spos: list[int] | None = None, mk: StartMarker | None = None,
                       chain: bool = False) -> torch.Tensor:
    """One plain forward, no cache, over the text up to the last read position followed by
    the markers (if mk) and the read tokens, under an explicit mask with the inline read's
    visibility and positions: the reference the cached batched read must equal, (K, d)."""
    T, K = max(pos), len(pos)
    M = K if mk is not None else 0
    e = model.get_input_embeddings()(torch.tensor([ids[:T]], device=dev))
    parts = [e] + ([mk.soft.to(model.dtype)[None].expand(1, K, -1)] if mk is not None else [])
    x = torch.cat(parts + [rt.soft.to(model.dtype)[None].expand(1, K, -1)], 1)
    n = T + M + K
    mask = torch.zeros(n, n, dtype=torch.bool, device=dev)
    mask[:T, :T] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
    position = list(range(T))
    if mk is not None:
        for k in range(K):
            mask[T + k, :spos[k]] = True
            mask[T + k, T + k] = True
        position += list(spos)
    for k in range(K):
        r = T + M + k
        mask[r, :pos[k]] = True
        mask[r, r] = True
        if mk is not None:
            mask[r, T + k] = True
        if chain:
            mask[r, T + M:r] = True
    position += list(pos)
    with read_hooks(model, rt, n=K), marker_hooks(model, mk, T, T + M):
        h = model.model(inputs_embeds=x, attention_mask=mask[None, None],
                        position_ids=torch.tensor([position], device=dev)).last_hidden_state
    return h[0, -K:].float()


def marker_reference(model, rt: ReadToken, mk: StartMarker, ids: list[int], pos: int, spos: int, dev):
    """uncached_reference for one read token and its marker, (d,)."""
    return uncached_reference(model, rt, ids, [pos], dev, [spos], mk)[0]


def seq_batches(lengths: list[int], n_items: list[int], max_tokens: int, max_items: int | None,
                shuffle_seed: int | None) -> list[list[int]]:
    """Length-sorted batches of sequences, at most max_tokens padded text tokens and
    max_items read tokens each (a single sequence always fits)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur, items = [], [], 0
    for i in order:
        fits = (len(cur) + 1) * lengths[i] <= max_tokens and (max_items is None or items + n_items[i] <= max_items)
        if cur and not fits:
            out.append(cur)
            cur, items = [], 0
        cur.append(i)
        items += n_items[i]
    if cur:
        out.append(cur)
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(out)
    return out


def row(seq: dict, mk: StartMarker | None) -> tuple:
    """The (ids, pos[, spos]) row text_pass takes for one built sequence."""
    return (seq["ids"], seq["pos"]) if mk is None else (seq["ids"], seq["pos"], seq["spos"])


def rth_items_per_step(tok, items: list[dict], max_tokens: int, max_seq_len: int) -> tuple[float, int]:
    """Items per optimizer step of RTH on the same items (verifier template), and its steps."""
    lens = [len(tokenize(tok, it)) for it in items]
    lens = [n for n in lens if n <= max_seq_len]
    n_steps = len(batches(lens, max_tokens, None))
    return len(lens) / n_steps, n_steps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["train", "encode"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layout", choices=sorted(LAYOUTS), default="chat")
    p.add_argument("--init_text", default=" correct")
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--start_marker", action="store_true", help="train: a start marker per step (H-S)")
    p.add_argument("--start_init_text", default=" step", help="train: the marker's starting embedding")
    p.add_argument("--start_deep", action="store_true", help="train: the marker gets its own offsets")
    p.add_argument("--chain", action="store_true",
                   help="train: a read token after every step, each seeing the earlier ones (H-C)")
    p.add_argument("--soft_attn", choices=["no_cudnn"], default=None)
    p.add_argument("--train", type=Path)
    p.add_argument("--n_train", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--max_tokens", type=int, default=16000)
    p.add_argument("--max_seq_len", type=int, default=3072, help="train: longer sequences are dropped")
    p.add_argument("--prm800k", nargs="*", default=[], help="encode: name:path[:limit]")
    p.add_argument("--pb", nargs="*", default=[], help="encode: subset:path")
    p.add_argument("--passive_blocks", default="24,35", help="encode: text-state blocks stored as `passive`")
    p.add_argument("--ckpt", type=Path, help="encode: read_token.pt (default: OUT/read_token.pt); "
                                             "an RTH token reads inline unchanged (Stage H1)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    from scripts.read_token_store import load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    d, n_layers = model.config.hidden_size, model.config.num_hidden_layers

    if a.phase == "train":
        emb = model.get_input_embeddings().weight
        tids = tok(a.init_text, add_special_tokens=False)["input_ids"]
        if len(tids) != 1:
            raise SystemExit(f"--init_text {a.init_text!r} is {len(tids)} tokens; inline reads use m = 1")
        init = emb[torch.tensor(tids, device=emb.device)].detach().float()
        rt = ReadToken(1, d, init.cpu(), head=a.head, n_layers=n_layers if a.deep else 0).to(dev)
        mk = None
        if a.start_marker:
            sids = tok(a.start_init_text, add_special_tokens=False)["input_ids"]
            if len(sids) != 1:
                raise SystemExit(f"--start_init_text {a.start_init_text!r} is {len(sids)} tokens, not 1")
            sinit = emb[torch.tensor(sids, device=emb.device)].detach().float().cpu()
            mk = StartMarker(d, sinit, n_layers if a.start_deep else 0).to(dev)
        items = prm800k_items(a.train, a.n_train, a.seed)
        if a.smoke:
            items = items[:a.smoke]
        per_step, rth_steps = rth_items_per_step(tok, items, a.max_tokens, a.max_seq_len)
        cap = math.ceil(per_step)
        chain = a.chain
        seqs, alone = build_sequences(tok, items, a.layout, chain=chain)
        n_long = sum(len(s["ids"]) > a.max_seq_len for s in seqs)
        seqs = [s for s in seqs if len(s["ids"]) <= a.max_seq_len]
        n_read = sum(len(s["pos"]) for s in seqs)
        bl = seq_batches([len(s["ids"]) for s in seqs], [len(s["members"]) for s in seqs],
                         a.max_tokens, cap, a.seed)
        print(f"[config] model={a.model} layout={a.layout} init={a.init_text!r} deep={a.deep} "
              f"head={a.head} n_train={a.n_train} seed={a.seed} lr={a.lr} start_marker="
              f"{repr(a.start_init_text) if mk is not None else None} start_deep={a.start_deep} chain={chain}", flush=True)
        print(f"[data] {len(items)} items -> {len(seqs)} sequences, {n_read} read slots "
              f"({alone} read alone, {n_long} sequences over {a.max_seq_len} tokens dropped); "
              f"{len(bl)} optimizer steps at <= {cap} items (RTH: {rth_steps} steps, "
              f"{per_step:.1f} items each)", flush=True)
        # Gate: one inline call must equal a read token appended to each standalone prefix
        # (with a marker or a chain: one uncached forward under the explicit mask).
        # Sequences with distinct read positions, so the gate covers several of them.
        with torch.no_grad():
            g = sorted(seqs[:500], key=lambda s: -len(set(s["pos"])))[:2]
            got = inline_read(model, rt, [row(s, mk) for s in g], pad, dev, grad=False,
                              soft_attn=a.soft_attn, mk=mk, chain=chain)
            k = 0
            for s in g:
                refs = (uncached_reference(model, rt, s["ids"], s["pos"], dev, s["spos"], mk, chain)
                        if mk is not None or chain else None)
                for j, pk in enumerate(s["pos"]):
                    ref = (read_all_states(model, rt, [s["ids"][:pk]], pad, dev, grad=False,
                                           soft_attn=a.soft_attn)[0, -1] if refs is None else refs[j])
                    rel = float((got[k] - ref).norm() / ref.norm())
                    print(f"[gate] read at {pk}: relative difference {rel:.4f}", flush=True)
                    if not rel <= 0.05:  # also catches NaN
                        raise SystemExit("inline read state does not match the appended read")
                    k += 1
        y = torch.tensor([float(items[i]["label"]) for s in seqs for i in s["members"]])
        offs = np.cumsum([0] + [len(s["members"]) for s in seqs])
        opt = make_optimizer(rt, a.lr, freeze_soft=False)
        if mk is not None:
            opt.add_param_group({"params": list(mk.parameters()), "lr": a.lr, "weight_decay": 0.0})
        params = [q for q in rt.parameters() if q.requires_grad] + (list(mk.parameters()) if mk else [])
        steps = a.epochs * len(bl)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
        t0, step, run = time.perf_counter(), 0, 0.0
        for ep in range(a.epochs):
            ebl = bl if ep == 0 else seq_batches([len(s["ids"]) for s in seqs],
                                                 [len(s["members"]) for s in seqs], a.max_tokens, cap,
                                                 a.seed + ep)
            for b in ebl:
                h = inline_read(model, rt, [row(seqs[i], mk) for i in b], pad, dev,
                                grad=True, soft_attn=a.soft_attn, mk=mk, chain=chain)
                h = h[member_index([seqs[i] for i in b], dev)]
                yb = torch.cat([y[offs[i]:offs[i + 1]] for i in b]).to(dev)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(rt.logit(h), yb)
                if not torch.isfinite(loss):
                    raise SystemExit(f"non-finite loss at step {step}")
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); step += 1
                run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
                if step % 50 == 0 or step == steps:
                    print(f"[train inline] step {step}/{steps} loss {run:.4f} "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)
        torch.save({"soft": rt.soft.detach().cpu(), "head": rt.head.state_dict(), "m": 1,
                    "model": a.model, "init": "text", "init_text": a.init_text, "read": "last",
                    "head_kind": a.head, "freeze_soft": False, "seed": a.seed, "n_train": a.n_train,
                    "soft_attn": a.soft_attn, "deep": None if rt.deep is None else rt.deep.detach().cpu(),
                    "interv": None, "rank": 0, "lora_dims": None, "interv_state": None, "lr": a.lr,
                    "layout": a.layout, "inline": True, "items_per_step": cap,
                    "start_soft": None if mk is None else mk.soft.detach().cpu(),
                    "start_deep": None if mk is None or mk.deep is None else mk.deep.detach().cpu(),
                    "start_init_text": a.start_init_text if mk is not None else None, "chain": chain},
                   a.out / "read_token.pt")
        return

    ck = torch.load(a.ckpt or a.out / "read_token.pt", map_location="cpu", weights_only=False)
    rt = read_token_from_ckpt(ck, model, dev)
    mk = marker_from_ckpt(ck, dev)
    chain = bool(ck.get("chain"))
    soft_attn = a.soft_attn or ck.get("soft_attn")
    pblocks = [int(x) for x in a.passive_blocks.split(",") if x]
    print(f"[encode] token {a.ckpt or a.out / 'read_token.pt'} (trained "
          f"{'inline, ' + ck['layout'] if ck.get('inline') else 'under the verifier template'}), "
          f"read inline in the {a.layout} layout, start marker: {mk is not None}, chain: {chain}", flush=True)
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
        t0 = time.perf_counter()
        seqs, alone = build_sequences(tok, items, a.layout, chain=chain)
        H = np.zeros((len(items), 1, d), np.float16)
        P = np.zeros((len(items), len(pblocks) + 1, d), np.float16)
        S = np.full(len(items), np.nan, np.float32)
        kept = np.zeros(len(items), bool)
        for b in seq_batches([len(s["ids"]) for s in seqs], [len(s["members"]) for s in seqs],
                             a.max_tokens, None, None):
            cap = {l: None for l in pblocks}
            cap["final"] = None
            with torch.no_grad():
                h = inline_read(model, rt, [row(seqs[i], mk) for i in b], pad, dev,
                                grad=False, soft_attn=soft_attn, passive=cap, mk=mk, chain=chain)
                mi = member_index([seqs[i] for i in b], dev)
                h = h[mi]
                cap = {k: v[mi] for k, v in cap.items()}
                s = torch.sigmoid(rt.logit(h)).cpu().numpy()
            if not torch.isfinite(h).all():
                raise SystemExit(f"non-finite read state in {name}")
            rows = [i for j in b for i in seqs[j]["members"]]
            H[rows, 0] = h.cpu().numpy()
            P[rows] = torch.stack([cap[l] for l in pblocks] + [cap["final"]], 1).cpu().numpy()
            S[rows] = s
            kept[rows] = True
        # one (n, 1, d) array per passive row: every array but `layers` must be per item
        passive = {f"passive_block{l}": P[:, j:j + 1] for j, l in enumerate(pblocks)}
        passive["passive_final"] = P[:, -1:]
        np.savez(a.out / f"{name}.shard0.npz", layers=np.array([-1]),
                 y=np.array([it["label"] for it in items], np.int8), kept=kept,
                 head_p_correct=1 - S, sep=H, **passive)
        with open(a.out / f"{name}.shard0.meta.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps({k: it[k] for k in ("uid", "id", "step_idx", "first_error", "label")
                                    if k in it}) + "\n")
        n_tok = sum(len(s["ids"]) + len(s["pos"]) * (2 if mk is not None else 1) for s in seqs)
        print(f"[encode] {name}: {int(kept.sum())}/{len(items)} items, {len(seqs)} sequences "
              f"({alone} read alone), {n_tok} tokens, in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
