#!/usr/bin/env python3
"""How fast the inline read token can score a trajectory, against the PRM (Stage H4).

On one GPU, the same N trajectories, every method timed on each trajectory in turn
(CUDA-synchronised), one trajectory per call (--batch 1) and then whole problems at a
time (--batch 10: the N = 10 candidates of a best-of-N problem in one call):

  text          the policy's forward over the trajectory in the chat layout (the pass
                generation already makes; the cache the read call reads)
  read_hf       the read call through the Hugging Face forward (read_token_inline.read_pass,
                the H3 margin)
  read_fast     the same call as a plain layer loop (scripts/inline_fast.FastReader)
  read_graph    FastReader captured as CUDA graphs (GraphedReader), cache copy included
  offline       text + read_graph: scoring a finished trajectory from scratch
  decode        one policy decode step at the trajectory's length (graphed layer loop,
                the unit of generation cost)
  decode_read   the same step with one read slot riding along (+ a marker slot for a
                marker token): the read piggybacked on generation, per step
  decode_hf     one decode step through the Hugging Face forward
  prm           Qwen2.5-Math-PRM-7B over its template, one forward (batched with padding)
  policy_on_prm_tokens  the policy's forward over the PRM's token ids (same length, other model)

Every fast read is checked against read_hf: relative state difference and P(incorrect)
difference per read.

    python scripts/analysis/inline_speed.py --inline_ckpt tokens/rthics_seed1/read_token.pt \\
        --trajectories <pool>/tts_math500.shard00_trajectories.jsonl --n 200 --out speed.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.rth_cost import PRM_SYSTEM  # noqa: E402
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from scripts.inline_fast import (  # noqa: E402
    MARKER, READ, TEXT, FastReader, FusedReader, GraphedReader, cache_kv)
from scripts.onpolicy.score_traces_with_read_token import read_jsonl, step_items  # noqa: E402
from scripts.read_token_inline import (  # noqa: E402
    build_sequences, marker_from_ckpt, read_pass, row, text_pass, uncached_reference)
from scripts.read_token_probe import soft_kernels  # noqa: E402
from scripts.read_token_store import load_model, load_read_token  # noqa: E402


def timed(f) -> float:
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        f()
    torch.cuda.synchronize()
    return time.perf_counter() - t


class Buckets(GraphedReader):
    """Slot buckets 1, 2, 4, 8, 16, then multiples of 16 (decode steps stay unpadded)."""

    def bucket(self, T: int, S: int) -> tuple[int, int]:
        Tb = -(-T // self.t_step) * self.t_step
        Sb = 1 << (S - 1).bit_length() if S <= 16 else -(-S // 16) * 16
        return Tb, Sb


def drop_after(cache, T: int) -> None:
    """Remove what a read or decode call appended (crop(-n) removes n in every version)."""
    n = cache.get_seq_length() - T
    if n > 0:
        cache.crop(-n)


def prm_batch(ptok, rows: list[dict], dev) -> tuple[torch.Tensor, torch.Tensor, int]:
    ids = []
    for r in rows:
        st = split_into_steps(r["solution"])
        text = ptok.apply_chat_template(
            [{"role": "system", "content": PRM_SYSTEM}, {"role": "user", "content": r["problem"]},
             {"role": "assistant", "content": "<extra_0>".join(st) + "<extra_0>"}],
            tokenize=False, add_generation_prompt=False)
        ids.append(ptok(text)["input_ids"])
    T = max(map(len, ids))
    pad = ptok.pad_token_id if ptok.pad_token_id is not None else ptok.eos_token_id
    x = torch.full((len(ids), T), pad, dtype=torch.long)
    att = torch.zeros((len(ids), T), dtype=torch.long)
    for b, t in enumerate(ids):
        x[b, :len(t)] = torch.tensor(t)
        att[b, :len(t)] = 1
    return x.to(dev), att.to(dev), sum(map(len, ids))


def decode_inputs(T: int, n_extra: int, marker: bool, dev):
    """One text slot (the next token) plus n_extra slots, all at position T over the whole
    cache; slots see the text and themselves, a read slot also its marker slot."""
    kinds = [TEXT] + ([MARKER, READ] if marker and n_extra else [READ] * n_extra)
    S = len(kinds)
    kind = torch.tensor([kinds], device=dev)
    mask = torch.zeros(1, 1, S, T + S, dtype=torch.bool, device=dev)
    mask[0, 0, :, :T] = True
    mask[0, 0, torch.arange(S), T + torch.arange(S)] = True
    if marker and n_extra:
        mask[0, 0, 2, T + 1] = True
    return kind, torch.full((1, S), T, device=dev), mask, torch.full((1, S), 0, device=dev)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--prm_model", default="Qwen/Qwen2.5-Math-PRM-7B")
    p.add_argument("--inline_ckpt", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 10])
    p.add_argument("--t_step", type=int, default=256)
    p.add_argument("--no_fused", action="store_true", help="skip the compiled FusedReader rows")
    p.add_argument("--cudnn", action="store_true", help="allow the cuDNN attention backend in the timed loop")
    p.add_argument("--order_check", type=int, default=30, help="trajectories for the call-order check (0: none)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    rows = [r for path in a.trajectories for r in read_jsonl(path)
            if r.get("gradeable") and (r.get("solution") or "").strip()]

    model, tok = load_model(a.model, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rt, ck = load_read_token(a.inline_ckpt, model.config.hidden_size, dev, model)
    mk = marker_from_ckpt(ck, dev)
    chain, sa = bool(ck.get("chain")), ck.get("soft_attn")
    pm, ptok = load_model(a.prm_model, dev)
    fr = FastReader(model, rt, mk)
    graphed = {"graph": Buckets(fr, t_step=a.t_step, kernels=lambda: soft_kernels(sa))}
    res: dict = {"gpu": torch.cuda.get_device_name(0), "ckpt": str(a.inline_ckpt), "chain": chain,
                 "marker": mk is not None, "t_step": a.t_step,
                 "attn_impl": {"policy": model.config._attn_implementation, "prm": pm.config._attn_implementation},
                 "torch": torch.__version__, "cudnn_attention": a.cudnn}
    if not a.no_fused:
        try:
            graphed["fused"] = Buckets(FusedReader(model, rt, mk), t_step=a.t_step, kernels=lambda: soft_kernels(sa))
        except Exception as e:  # torch.compile unavailable: report the graphed loop only
            res["fused_error"] = repr(e)
            print(f"[fused] disabled: {e!r}", flush=True)

    def seqs_of(rs):
        return [row(q, mk) for r in rs for q in build_sequences(tok, step_items(r), "chat", chain=chain)[0]]

    def p_incorrect(h):
        return torch.sigmoid(rt.logit(h).double())

    # every forward without the cuDNN attention backend (flash, memory-efficient or math):
    # in the 650971 smoke the first call at a new sequence length took 2-3x longer than a
    # repeat, which would bill the PRM for a per-length setup cost; the order check
    # measures both settings on unseen lengths
    with (contextlib.nullcontext() if a.cudnn else soft_kernels("no_cudnn")):
        for B in a.batch:
            if B == 1:
                groups = [[r] for r in rows[:a.n + a.warmup]]
            else:
                by = defaultdict(list)
                for r in rows:
                    by[r["fork_id"]].append(r)
                groups = [g[:B] for g in by.values() if len(g) >= B][:max(1, (a.n + a.warmup) // B)]
            warm = a.warmup if B == 1 else 1
            gseqs = [seqs_of(g) for g in groups]
            need = defaultdict(lambda: [0, 3])  # rows -> [text, slots]; a decode step holds up to 3
            for sq in gseqs:
                nd = need[len(sq)]
                nd[0] = max(nd[0], max(len(s[0]) for s in sq))
                nd[1] = max(nd[1], max(len(s[1]) for s in sq) * (2 if mk is not None else 1))
            for gr in graphed.values():
                for rows_b, (t, n_s) in need.items():
                    if rows_b not in gr.kv:
                        gr.reserve(rows_b, t, n_s)
            # capture every bucket first, so no timed call includes a capture (or a compile)
            for gname in list(graphed):
                gr = graphed[gname]
                t0 = time.perf_counter()
                try:
                    for g, seqs in zip(groups, gseqs):
                        with torch.no_grad():
                            cache, lay = text_pass(model, seqs, pad, dev, marker=mk is not None, chain=chain)
                            gr.read(cache, lay)
                            if B == 1:
                                T = cache.get_seq_length()
                                ks, vs = cache_kv(cache)
                                for n_extra in (0, 1):
                                    kind, pos, mask, ids = decode_inputs(T, n_extra, mk is not None, dev)
                                    gr.run(ks, vs, kind, pos, mask, ids)
                except Exception as e:
                    if gname == "graph":
                        raise
                    res["fused_error"] = repr(e)
                    print(f"[fused] disabled: {e!r}", flush=True)
                    del graphed[gname]
                    continue
                res[f"b{B}_capture_{gname}"] = {"graphs": len(gr.graphs), "sec": time.perf_counter() - t0}
                print(f"[b{B}] {len(groups)} groups, {gname}: {len(gr.graphs)} graphs captured in "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)

            sec, toks = defaultdict(list), defaultdict(list)
            n_tr, rel, dp = [], [], []
            for i, (g, seqs) in enumerate(zip(groups, gseqs)):
                hold = {}
                t_text = timed(lambda: hold.update(cl=text_pass(model, seqs, pad, dev, marker=mk is not None,
                                                                chain=chain)))
                cache, lay = hold["cl"]
                T = cache.get_seq_length()
                t_hf = timed(lambda: hold.update(hf=read_pass(model, rt, cache, lay, grad=False, soft_attn=sa, mk=mk)))
                drop_after(cache, T)
                with soft_kernels(sa):
                    t_fast = timed(lambda: hold.update(fast=fr.read(cache, lay)))
                per = {"text": t_text, "read_hf": t_hf, "read_fast": t_fast}
                for gname, gr in graphed.items():
                    per[f"read_{gname}"] = timed(lambda: hold.update({gname: gr.read(cache, lay)}))
                    per[f"offline_{gname}"] = t_text + per[f"read_{gname}"]
                if B == 1:
                    ks, vs = cache_kv(cache)
                    for gname, gr in graphed.items():
                        for n_extra, name in ((0, "decode"), (1, "decode_read")):
                            kind, pos, mask, ids = decode_inputs(T, n_extra, mk is not None, dev)
                            per[f"{name}_{gname}"] = timed(lambda: gr.run(ks, vs, kind, pos, mask, ids))
                    # the bf16 noise floor: the same reads as one uncached forward per sequence
                    hold["uncached"] = torch.cat([
                        uncached_reference(model, rt, s[0], s[1], dev, s[2] if mk is not None else None, mk, chain)
                        for s in seqs])
                    nxt = torch.tensor([[pad]], device=dev)
                    per["decode_hf"] = timed(lambda: model.model(input_ids=nxt, past_key_values=cache, use_cache=True))
                    drop_after(cache, T)
                x, att, n_prm = prm_batch(ptok, g, dev)
                per["prm"] = timed(lambda: pm.model(input_ids=x, attention_mask=att))
                # the policy over the PRM's own tokens: separates the model from the input length
                per["policy_on_prm_tokens"] = timed(lambda: model.model(input_ids=x, attention_mask=att))
                with torch.no_grad():
                    for name in [n for n in ("fast", *graphed, "uncached") if n in hold]:
                        d = (hold[name] - hold["hf"]).norm(dim=-1) / hold["hf"].norm(dim=-1)
                        rel.append((name, d.cpu().numpy()))
                        dp.append((name, (p_incorrect(hold[name]) - p_incorrect(hold["hf"])).abs().cpu().numpy()))
                if i >= warm:
                    n_tr.append(len(g))
                    for k, v in per.items():
                        sec[k].append(v)
                    n_text = sum(len(s[0]) for s in seqs)
                    toks["text"].append(n_text)
                    toks["read"].append(int(len(lay["rows"])) * (2 if mk is not None else 1))
                    toks["prm"].append(n_prm)
            out = {}
            for k, v in sec.items():
                pt = np.array(v) / np.array(n_tr)
                out[k] = {"sec_per_trace_mean": float(pt.mean()), "sec_per_trace_median": float(np.median(pt)),
                          "sec_per_trace_p90": float(np.percentile(pt, 90)), "calls": len(v)}
            out["tokens_per_trace"] = {k: float(np.sum(v) / np.sum(n_tr)) for k, v in toks.items()}
            out["equivalence_vs_read_hf"] = {}
            for n in dict.fromkeys(m for m, _ in rel):
                r_ = np.concatenate([x for m, x in rel if m == n])
                p_ = np.concatenate([x for m, x in dp if m == n])
                out["equivalence_vs_read_hf"][n] = {"reads": int(len(r_)), "rel_state_diff_mean": float(r_.mean()),
                                                    "rel_state_diff_max": float(r_.max()),
                                                    "p_diff_mean": float(p_.mean()), "p_diff_max": float(p_.max())}
            res[f"b{B}"] = out
            print(f"[b{B}] traces {int(np.sum(n_tr))}, per-trace seconds (mean / median):", flush=True)
            for k, v in out.items():
                if "sec_per_trace_mean" in v:
                    print(f"  {k:12s} {v['sec_per_trace_mean'] * 1e3:8.2f} ms  {v['sec_per_trace_median'] * 1e3:8.2f} ms",
                          flush=True)
            print(f"  tokens {out['tokens_per_trace']}", flush=True)
            for n, v in out["equivalence_vs_read_hf"].items():
                print(f"  vs read_hf {n:9s} {v}", flush=True)
    if a.order_check:
        # the same calls in a fixed rotation, to see whether a call's place in the loop moves it
        # on trajectories no earlier call has seen (the end of the file), so each length is new
        order = ["prm", "policy_on_prm_tokens", "text", "policy_on_prm_tokens", "prm", "text"]
        k = a.order_check
        for setting, sub in (("default", rows[-2 * k:-k]), ("no_cudnn", rows[-k:])):
            oc = defaultdict(list)
            with (soft_kernels("no_cudnn") if setting == "no_cudnn" else contextlib.nullcontext()):
                for r in sub:
                    seqs = seqs_of([r])
                    x, att, _ = prm_batch(ptok, [r], dev)
                    for j, name in enumerate(order):
                        f = {"prm": lambda: pm.model(input_ids=x, attention_mask=att),
                             "policy_on_prm_tokens": lambda: model.model(input_ids=x, attention_mask=att),
                             "text": lambda: text_pass(model, seqs, pad, dev, marker=mk is not None, chain=chain)}[name]
                        oc[f"{j}_{name}"].append(timed(f))
            res[f"order_check_{setting}"] = {kk: float(np.median(v)) for kk, v in oc.items()}
            print(f"[order {setting}] medians (ms):",
                  {kk: round(v * 1e3, 2) for kk, v in res[f"order_check_{setting}"].items()}, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
