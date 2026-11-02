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

With --reprobe_release (a released ReProbe hidden-state probe, reprobe_comparison_v2), on
the policy's final hidden states of the same text (what generation already computes, so
they are given, as the read token is given the cache):

  reprobe_head     ReProbe's head over every step of the trajectory (one copy of the
                   trace per step, as it scores a finished trace)
  reprobe_offline  text + reprobe_head
  reprobe_step     the head for the last step only: its per-step cost during generation
                   (its encoder sees the whole trace with a per-step entity embedding, so
                   nothing carries over from the previous step); batch 1 only
  reprobe_head_graph, reprobe_step_graph
                   the same calls captured as CUDA graphs (text length padded to a
                   multiple of --t_step, input copy included), the counterpart of read_graph

Step spans for the head are the text between consecutive read positions (timing only).

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


def reprobe_inputs(model, seqs: list[tuple], pad: int, dev) -> dict:
    """ReProbe head inputs for the rows of `seqs`: final hidden states, valid positions, and
    the claim masks of every step and of each row's last step. Step k of a row spans the
    text tokens [pos[k-1], pos[k]) (k = 0: the token before pos[0]), read at feature p - 1."""
    T = max(len(r[0]) for r in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    claims, owner, last = [], [], []
    for b, r in enumerate(seqs):
        ids[b, :len(r[0])] = torch.tensor(r[0])
        att[b, :len(r[0])] = 1
        for k, pk in enumerate(r[1]):
            lo = r[1][k - 1] if k else pk - 1
            m = torch.zeros(T, dtype=torch.long)
            m[max(lo - 1, 0):max(pk - 1, 1)] = 1
            claims.append(m)
            owner.append(b)
        last.append(len(claims) - 1)
    ids, att = ids.to(dev), att.to(dev)
    with torch.no_grad():
        feats = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    claims, owner = torch.stack(claims).to(dev), torch.tensor(owner, device=dev)
    return {"feats": feats, "valid": att.bool(), "claims": claims, "owner": owner,
            "claims_last": claims[last], "owner_last": owner[last]}


class GraphedHead:
    """ReProbe's head as one CUDA graph per (rows, claims, padded length): padding positions
    are invalid, so they are never attended and the logits are unchanged."""

    def __init__(self, head, run_head, t_step: int):
        self.head, self.run_head, self.t_step, self.graphs = head, run_head, t_step, {}
        self.pool = None

    def call(self, feats, valid, claims, owner):
        B, T, F = feats.shape
        Tb = -(-T // self.t_step) * self.t_step
        key = (B, claims.shape[0], Tb)
        if key not in self.graphs:
            st = {"feats": torch.zeros(B, Tb, F, dtype=feats.dtype, device=feats.device),
                  "valid": torch.zeros(B, Tb, dtype=torch.bool, device=feats.device),
                  "claims": torch.zeros(claims.shape[0], Tb, dtype=claims.dtype, device=feats.device),
                  "owner": torch.zeros_like(owner)}
            f = lambda: self.run_head(self.head, st["feats"], st["valid"], st["claims"], st["owner"], "visible")
            st["valid"][:, 0] = True
            st["claims"][:, 0] = 1
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(2):
                    f()
            torch.cuda.current_stream().wait_stream(s)
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, pool=self.pool):
                st["out"] = f()
            self.graphs[key] = (g, st)
        g, st = self.graphs[key]

        def run():
            st["feats"][:, :T].copy_(feats)
            st["valid"].zero_()
            st["valid"][:, :T].copy_(valid)
            st["claims"].zero_()
            st["claims"][:, :T].copy_(claims)
            st["owner"].copy_(owner)
            g.replay()
            return st["out"]
        return run


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
    p.add_argument("--reprobe_release", type=Path, default=None,
                   help="a released ReProbe hidden-state probe directory: also time its head")
    p.add_argument("--reprobe_claim_tokens", type=int, default=160000,
                   help="steps x length per head call, as reprobe_probe.score_traces micro-batches")
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
    rp_head = rp_graph = None
    if a.reprobe_release is not None:
        from scripts.reprobe_cmp.reprobe_probe import head_from_release, run_head
        from src.probes.reprobe_head import n_params
        rp_head = head_from_release(a.reprobe_release, dev, model.config.hidden_size)
        rp_graph = GraphedHead(rp_head, run_head, a.t_step)
    fr = FastReader(model, rt, mk)
    graphed = {"graph": Buckets(fr, t_step=a.t_step, kernels=lambda: soft_kernels(sa))}
    res: dict = {"gpu": torch.cuda.get_device_name(0), "ckpt": str(a.inline_ckpt), "chain": chain,
                 "marker": mk is not None, "t_step": a.t_step,
                 "attn_impl": {"policy": model.config._attn_implementation, "prm": pm.config._attn_implementation},
                 "torch": torch.__version__, "cudnn_attention": a.cudnn,
                 "params": {"read_token": sum(p.numel() for p in rt.parameters())
                            + (sum(p.numel() for p in mk.parameters()) if mk is not None else 0)}}
    if rp_head is not None:
        res["reprobe_release"] = str(a.reprobe_release)
        res["params"]["reprobe_head"] = n_params(rp_head)
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
            n_tr, rel, dp, rp_eq = [], [], [], []
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
                if rp_head is not None:
                    ri = reprobe_inputs(model, seqs, pad, dev)
                    k = max(1, a.reprobe_claim_tokens // ri["feats"].shape[1])
                    calls = {"reprobe_head": [(ri["claims"][c:c + k], ri["owner"][c:c + k])
                                              for c in range(0, len(ri["owner"]), k)]}
                    if B == 1:
                        calls["reprobe_step"] = [(ri["claims_last"], ri["owner_last"])]
                    eager = {n: [lambda c=c, o=o: run_head(rp_head, ri["feats"], ri["valid"], c, o, "visible")
                                 for c, o in cl] for n, cl in calls.items()}
                    for n, fs in eager.items():
                        per[n] = timed(lambda fs=fs: [f() for f in fs])
                    per["reprobe_offline"] = t_text + per["reprobe_head"]
                    if rp_graph is not None:
                        try:
                            for n, cl in calls.items():
                                fs = [rp_graph.call(ri["feats"], ri["valid"], c, o) for c, o in cl]
                                for f in fs:  # captures outside the timed call
                                    f()
                                per[n + "_graph"] = timed(lambda fs=fs: [f() for f in fs])
                                with torch.no_grad():
                                    for f, ef in zip(fs, eager[n]):
                                        rp_eq.append((n + "_graph", (torch.sigmoid(f().double())
                                                                     - torch.sigmoid(ef().double())).abs().cpu().numpy()))
                        except Exception as e:  # capture unsupported: keep the eager rows
                            res["reprobe_graph_error"] = repr(e)
                            print(f"[reprobe graph] disabled: {e!r}", flush=True)
                            rp_graph = None
                    del ri, calls, eager
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
            for n in dict.fromkeys(m for m, _ in rp_eq):
                p_ = np.concatenate([x for m, x in rp_eq if m == n])
                out["equivalence_vs_read_hf"][n + "_vs_eager"] = {"reads": int(len(p_)), "p_diff_mean": float(p_.mean()),
                                                                  "p_diff_max": float(p_.max())}
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
