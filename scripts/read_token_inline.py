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
import json
import math
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


def encode_text(tok, text: str) -> tuple[list[int], dict[int, int]]:
    """Token ids and {character end: number of tokens up to the last token ending there}."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ends = {}
    for j, (_, e) in enumerate(enc["offset_mapping"]):
        ends[e] = j + 1
    return list(enc["input_ids"]), ends


def build_sequences(tok, items: list[dict], layout: str = "chat") -> tuple[list[dict], int]:
    """Tokenised sequences {"ids", "pos", "members"} (pos[k]: text tokens member k reads),
    and how many items had to be read on their own (no token boundary at their separator)."""
    prompt_of = LAYOUTS[layout]
    out, alone = [], 0
    for g in group_items(items):
        prompt = prompt_of(g["problem"])
        ids, ends = encode_text(tok, prompt + g["text"] + SEP)
        seq = {"ids": ids, "pos": [], "members": []}
        for i in g["members"]:
            p = ends.get(len(prompt) + len(item_text(items[i])) + len(SEP))
            if p is None:
                alone += 1
                own, _ = encode_text(tok, prompt + item_text(items[i]) + SEP)
                out.append({"ids": own, "pos": [len(own)], "members": [i]})
                continue
            seq["pos"].append(p)
            seq["members"].append(i)
        if seq["members"]:
            order = np.argsort(seq["pos"], kind="stable")
            seq["pos"] = [seq["pos"][k] for k in order]
            seq["members"] = [seq["members"][k] for k in order]
            out.append(seq)
    return out, alone


def gather_hooks(model, blocks: list[int], rows: torch.Tensor, cols: torch.Tensor, out: dict):
    """Record decoder block l's output at positions (rows, cols) into out[l], (n, d)."""
    def make(l):
        def hook(_mod, _inp, o):
            h = o[0] if isinstance(o, tuple) else o
            out[l] = h[rows, cols].float()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make(l)) for l in blocks]
    return handles


def text_pass(model, seqs: list[tuple[list[int], list[int]]], pad: int, dev,
              passive: dict | None = None) -> tuple[object, dict]:
    """The text of every row, once, without gradients: (its cache, the read layout).

    seqs: (text ids, pos) per row, pos[k] the number of text tokens read token k sees.
    passive: {block: None, "final": None} to also fill in the text states at token
    pos[k] - 1 (the separator the read token follows), block outputs and post-norm.
    """
    B = len(seqs)
    T = max(len(ids) for ids, _ in seqs)
    K = max(len(p) for _, p in seqs)
    ids = torch.full((B, T), pad, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    mask = torch.zeros((B, 1, K, T + K), dtype=torch.bool)
    pos2 = torch.zeros((B, K), dtype=torch.long)
    rows, cols = [], []
    for b, (t, p) in enumerate(seqs):
        if not p or max(p) > len(t) or min(p) < 1:
            raise ValueError(f"row {b}: read positions {p} outside 1..{len(t)}")
        ids[b, :len(t)] = torch.tensor(t)
        att[b, :len(t)] = 1
        for k, pk in enumerate(p):
            mask[b, 0, k, :pk] = True
            pos2[b, k] = pk
            rows.append(b)
            cols.append(k)
        mask[b, 0, torch.arange(K), T + torch.arange(K)] = True  # every slot sees itself
    rows, cols = torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)
    tcols = pos2.to(dev)[rows, cols] - 1
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
    return out.past_key_values, {"mask": mask.to(dev), "pos": pos2.to(dev), "rows": rows, "cols": cols}


def read_pass(model, rt: ReadToken, cache, lay: dict, grad: bool, soft_attn: str | None = None) -> torch.Tensor:
    """Every read token over a text_pass cache in one call, (sum of K, d). Appends to the cache."""
    if rt.m != 1:
        raise ValueError("inline reads need an m = 1 read token")
    B, K = lay["pos"].shape
    emb = rt.soft.to(model.dtype)[None].expand(B, K, -1)
    with torch.set_grad_enabled(grad), read_hooks(model, rt, n=K), soft_kernels(soft_attn):
        h = model.model(inputs_embeds=emb, attention_mask=lay["mask"], position_ids=lay["pos"],
                        past_key_values=cache, use_cache=True).last_hidden_state
    return h.float()[lay["rows"], lay["cols"]]


def inline_read(model, rt: ReadToken, seqs: list[tuple[list[int], list[int]]], pad: int, dev,
                grad: bool, soft_attn: str | None = None, passive: dict | None = None) -> torch.Tensor:
    """Final-layer (post-norm) states of every read token, (sum of K, d), row by row
    (text_pass, then read_pass)."""
    if rt.m != 1:
        raise ValueError("inline reads need an m = 1 read token")
    cache, lay = text_pass(model, seqs, pad, dev, passive)
    return read_pass(model, rt, cache, lay, grad, soft_attn)


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
        items = prm800k_items(a.train, a.n_train, a.seed)
        if a.smoke:
            items = items[:a.smoke]
        per_step, rth_steps = rth_items_per_step(tok, items, a.max_tokens, a.max_seq_len)
        cap = math.ceil(per_step)
        seqs, alone = build_sequences(tok, items, a.layout)
        n_long = sum(len(s["ids"]) > a.max_seq_len for s in seqs)
        seqs = [s for s in seqs if len(s["ids"]) <= a.max_seq_len]
        n_read = sum(len(s["members"]) for s in seqs)
        bl = seq_batches([len(s["ids"]) for s in seqs], [len(s["members"]) for s in seqs],
                         a.max_tokens, cap, a.seed)
        print(f"[config] model={a.model} layout={a.layout} init={a.init_text!r} deep={a.deep} "
              f"head={a.head} n_train={a.n_train} seed={a.seed} lr={a.lr}", flush=True)
        print(f"[data] {len(items)} items -> {len(seqs)} sequences, {n_read} read tokens "
              f"({alone} read alone, {n_long} sequences over {a.max_seq_len} tokens dropped); "
              f"{len(bl)} optimizer steps at <= {cap} items (RTH: {rth_steps} steps, "
              f"{per_step:.1f} items each)", flush=True)
        # Gate: one inline call must equal a read token appended to each standalone prefix.
        with torch.no_grad():
            g = sorted(seqs[:200], key=lambda s: -len(s["pos"]))[:2]
            got = inline_read(model, rt, [(s["ids"], s["pos"]) for s in g], pad, dev, grad=False,
                              soft_attn=a.soft_attn)
            k = 0
            for s in g:
                for pk in s["pos"]:
                    ref = read_all_states(model, rt, [s["ids"][:pk]], pad, dev, grad=False,
                                          soft_attn=a.soft_attn)[0, -1]
                    rel = float((got[k] - ref).norm() / ref.norm())
                    print(f"[gate] read at {pk}: relative difference {rel:.4f}", flush=True)
                    if not rel <= 0.05:  # also catches NaN
                        raise SystemExit("inline read state does not match the appended read")
                    k += 1
        y = torch.tensor([float(items[i]["label"]) for s in seqs for i in s["members"]])
        offs = np.cumsum([0] + [len(s["members"]) for s in seqs])
        opt = make_optimizer(rt, a.lr, freeze_soft=False)
        params = [q for q in rt.parameters() if q.requires_grad]
        steps = a.epochs * len(bl)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
        t0, step, run = time.perf_counter(), 0, 0.0
        for ep in range(a.epochs):
            ebl = bl if ep == 0 else seq_batches([len(s["ids"]) for s in seqs],
                                                 [len(s["members"]) for s in seqs], a.max_tokens, cap,
                                                 a.seed + ep)
            for b in ebl:
                h = inline_read(model, rt, [(seqs[i]["ids"], seqs[i]["pos"]) for i in b], pad, dev,
                                grad=True, soft_attn=a.soft_attn)
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
                    "layout": a.layout, "inline": True, "items_per_step": cap},
                   a.out / "read_token.pt")
        return

    ck = torch.load(a.ckpt or a.out / "read_token.pt", map_location="cpu", weights_only=False)
    rt = read_token_from_ckpt(ck, model, dev)
    soft_attn = a.soft_attn or ck.get("soft_attn")
    pblocks = [int(x) for x in a.passive_blocks.split(",") if x]
    print(f"[encode] token {a.ckpt or a.out / 'read_token.pt'} (trained "
          f"{'inline, ' + ck['layout'] if ck.get('inline') else 'under the verifier template'}), "
          f"read inline in the {a.layout} layout", flush=True)
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
        seqs, alone = build_sequences(tok, items, a.layout)
        H = np.zeros((len(items), 1, d), np.float16)
        P = np.zeros((len(items), len(pblocks) + 1, d), np.float16)
        S = np.full(len(items), np.nan, np.float32)
        kept = np.zeros(len(items), bool)
        for b in seq_batches([len(s["ids"]) for s in seqs], [len(s["members"]) for s in seqs],
                             a.max_tokens, None, None):
            cap = {l: None for l in pblocks}
            cap["final"] = None
            with torch.no_grad():
                h = inline_read(model, rt, [(seqs[i]["ids"], seqs[i]["pos"]) for i in b], pad, dev,
                                grad=False, soft_attn=soft_attn, passive=cap)
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
        n_tok = sum(len(s["ids"]) + len(s["pos"]) for s in seqs)
        print(f"[encode] {name}: {int(kept.sum())}/{len(items)} items, {len(seqs)} sequences "
              f"({alone} read alone), {n_tok} tokens, in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
