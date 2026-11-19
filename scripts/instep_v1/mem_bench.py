#!/usr/bin/env python3
"""Peak GPU memory of InStep, the PRM and the ReProbe head on the speed benchmark's trajectories
(reviewer Q3 / W4). Same trajectories, groups and kernels as scripts/analysis/inline_speed.py
(MATH-500 pool, --n trajectories, one call per trajectory or per problem's first B candidates,
cuDNN attention off); one --mode per process, so nothing another mode allocated is counted.

Every number is torch.cuda.max_memory_allocated() (GiB) after reset_peak_memory_stats(), per call,
then the max and mean over the calls; `resident` is what stays allocated between calls (weights,
captured graphs and their static buffers).

  policy   Qwen3-8B alone, the reference without a verifier.
           prefill   forward over the candidates' text with its key/value cache (what generation
                     holds once the candidates are written; the cache the step token reads)
           decode    one decode step with that cache alive (generation's steady state)
           kv_cache  bytes of the candidates' cache (padded batch), and the unpadded value
  instep   Qwen3-8B + the InStep token.
           online_hf     the step-token pass over the live cache (Hugging Face forward, no copy)
           online_fast   the same pass as a plain layer loop reading the cache (FastReader, no copy)
  instep_fused  the same, plus online_fused: the timed reader of the latency table (compiled, CUDA
                graphs), whose static buffers hold a copy of the cache (fused_static, in resident);
                one batch size per process
           side_cache    bytes the step tokens and markers add to the cache (they stay there)
           offline       text pass + step-token pass from nothing (HF forward)
  prm      Qwen2.5-Math-PRM-7B's scoring forward (no key/value cache kept) with the policy's weights
           resident (prm), and with the candidates' policy cache alive too (prm_with_policy_cache)
  reprobe  Qwen3-8B + ReProbe's released head: the policy's final states over the text (kept for
           the head), then the head over every step (micro-batched as when it scores a trace)

    python scripts/instep_v1/mem_bench.py --mode instep --inline_ckpt <read_token.pt> \\
        --trajectories <pool>.jsonl --n 200 --batch 1 5 10 --out mem_instep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.inline_speed import Buckets, drop_after, prm_batch, reprobe_inputs  # noqa: E402
from scripts.onpolicy.score_traces_with_read_token import read_jsonl, step_items  # noqa: E402
from scripts.read_token_inline import build_sequences, marker_from_ckpt, read_pass, row, text_pass  # noqa: E402
from scripts.read_token_probe import soft_kernels  # noqa: E402
from scripts.read_token_store import load_model, load_read_token  # noqa: E402

GIB = 2 ** 30


def groups_of(rows: list[dict], B: int, n: int) -> list[list[dict]]:
    """inline_speed.py's groups, without its warm-up calls."""
    if B == 1:
        return [[r] for r in rows[:n]]
    by = defaultdict(list)
    for r in rows:
        by[r["fork_id"]].append(r)
    return [g[:B] for g in by.values() if len(g) >= B][:max(1, n // B)]


def kv_bytes(cache) -> int:
    return int(sum(l.keys.numel() * l.keys.element_size() + l.values.numel() * l.values.element_size()
                   for l in cache.layers))


def peak(f) -> tuple[float, object]:
    """(peak GiB allocated during f, f's result)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = f()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / GIB, out


def summary(v: list[float]) -> dict:
    return {"max": float(np.max(v)), "mean": float(np.mean(v)), "calls": len(v)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["policy", "instep", "instep_fused", "prm", "reprobe"], required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--prm_model", default="Qwen/Qwen2.5-Math-PRM-7B")
    p.add_argument("--inline_ckpt", type=Path, required=True)
    p.add_argument("--reprobe_release", type=Path, default=None)
    p.add_argument("--reprobe_claim_tokens", type=int, default=160000)
    p.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--t_step", type=int, default=256)
    p.add_argument("--no_fused", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    rows = [r for path in a.trajectories for r in read_jsonl(path)
            if r.get("gradeable") and (r.get("solution") or "").strip()]
    res: dict = {"mode": a.mode, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                 "n": a.n, "unit": "GiB", "ckpt": str(a.inline_ckpt)}
    pm = ptok = head = None
    # the policy is loaded in every mode: its tokenizer builds the trajectories' sequences, and the
    # PRM and ReProbe modes also measure with the policy resident
    model, tok = load_model(a.model, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    res["weights_policy"] = torch.cuda.memory_allocated() / GIB
    rt, ck = load_read_token(a.inline_ckpt, model.config.hidden_size, dev, model)
    mk = marker_from_ckpt(ck, dev)
    chain, sa = bool(ck.get("chain")), ck.get("soft_attn")
    res["weights_token"] = torch.cuda.memory_allocated() / GIB - res["weights_policy"]
    if a.mode == "prm":
        w0 = torch.cuda.memory_allocated() / GIB
        pm, ptok = load_model(a.prm_model, dev)
        res["weights_prm"] = torch.cuda.memory_allocated() / GIB - w0
    if a.mode == "reprobe":
        from scripts.reprobe_cmp.reprobe_probe import head_from_release, run_head
        w0 = torch.cuda.memory_allocated() / GIB
        head = head_from_release(a.reprobe_release, dev, model.config.hidden_size)
        res["weights_reprobe_head"] = torch.cuda.memory_allocated() / GIB - w0
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    per_tok = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * hd * 2  # bf16 keys + values
    res["kv_bytes_per_token"] = per_tok

    def seqs_of(rs):
        return [row(q, mk) for r in rs for q in build_sequences(tok, step_items(r), "chat", chain=chain)[0]]

    # the compiled, graphed reader keeps static buffers per bucket (a copy of the cache among them):
    # measured in its own mode and process (instep_fused, one batch size per run), so they never
    # sit under the other rows' peaks
    fused = None
    if a.mode in ("instep", "instep_fused"):
        from scripts.inline_fast import FastReader, FusedReader
        fr = FastReader(model, rt, mk)
        if a.mode == "instep_fused":
            if len(a.batch) != 1:
                raise SystemExit("instep_fused: one --batch per process")
            fused = Buckets(FusedReader(model, rt, mk), t_step=a.t_step, kernels=lambda: soft_kernels(sa))
    with soft_kernels("no_cudnn"):
        for B in a.batch:
            groups = groups_of(rows, B, a.n)
            gseqs = [seqs_of(g) for g in groups]
            m = defaultdict(list)
            if fused is not None:
                need = defaultdict(lambda: [0, 3])
                for sq in gseqs:
                    nd = need[len(sq)]
                    nd[0] = max(nd[0], max(len(s[0]) for s in sq))
                    nd[1] = max(nd[1], max(len(s[1]) for s in sq) * (2 if mk is not None else 1))
                before = torch.cuda.memory_allocated() / GIB
                for rows_b, (t, n_s) in need.items():
                    if rows_b not in fused.kv:
                        fused.reserve(rows_b, t, n_s)
                for seqs in gseqs:  # capture every bucket outside the measured calls
                    with torch.no_grad():
                        cache, lay = text_pass(model, seqs, pad, dev, marker=mk is not None, chain=chain)
                        fused.read(cache, lay)
                    del cache, lay
                torch.cuda.empty_cache()
                res[f"b{B}_fused_static"] = torch.cuda.memory_allocated() / GIB - before
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            resident = torch.cuda.memory_allocated() / GIB
            for g, seqs in zip(groups, gseqs):
                n_text = sum(len(s[0]) for s in seqs)
                if a.mode in ("policy", "instep", "instep_fused"):
                    pk, (cache, lay) = peak(lambda: text_pass(model, seqs, pad, dev, marker=mk is not None,
                                                              chain=chain))
                    m["prefill"].append(pk)
                    m["kv_cache_padded"].append(kv_bytes(cache) / GIB)
                    m["kv_cache_unpadded"].append(n_text * per_tok / GIB)
                    m["with_cache"].append(torch.cuda.memory_allocated() / GIB)
                    T = cache.get_seq_length()
                    if a.mode == "policy":
                        nxt = torch.full((len(seqs), 1), pad, device=dev)
                        pk, o = peak(lambda: model.model(input_ids=nxt, past_key_values=cache, use_cache=True))
                        m["decode"].append(pk)
                        del o, nxt  # the output holds the cache
                        drop_after(cache, T)
                    else:
                        kv0 = kv_bytes(cache)
                        pk, _ = peak(lambda: read_pass(model, rt, cache, lay, grad=False, soft_attn=sa, mk=mk))
                        m["online_hf"].append(pk)
                        m["side_cache"].append((kv_bytes(cache) - kv0) / GIB)
                        m["side_cache_unpadded"].append(
                            int(len(lay["rows"])) * (2 if mk is not None else 1) * per_tok / GIB)
                        drop_after(cache, T)
                        with soft_kernels(sa):
                            pk, _ = peak(lambda: fr.read(cache, lay))
                        m["online_fast"].append(pk)
                        if fused is not None:
                            pk, _ = peak(lambda: fused.read(cache, lay))
                            m["online_fused"].append(pk)
                    del cache, lay
                    torch.cuda.empty_cache()
                    if a.mode in ("instep", "instep_fused"):
                        def offline():
                            c, l_ = text_pass(model, seqs, pad, dev, marker=mk is not None, chain=chain)
                            return read_pass(model, rt, c, l_, grad=False, soft_attn=sa, mk=mk)
                        pk, _ = peak(offline)
                        m["offline_hf"].append(pk)
                        torch.cuda.empty_cache()
                elif a.mode == "prm":
                    x, att, _ = prm_batch(ptok, g, dev)
                    # a scoring forward keeps no key/value cache (use_cache=False)
                    pk, o = peak(lambda: pm.model(input_ids=x, attention_mask=att, use_cache=False))
                    m["prm"].append(pk)
                    del o
                    _, (cache, lay) = peak(lambda: text_pass(model, seqs, pad, dev, marker=mk is not None,
                                                             chain=chain))
                    pk, o = peak(lambda: pm.model(input_ids=x, attention_mask=att, use_cache=False))
                    m["prm_with_policy_cache"].append(pk)
                    del cache, lay, x, att, o
                    torch.cuda.empty_cache()
                else:
                    pk, ri = peak(lambda: reprobe_inputs(model, seqs, pad, dev))
                    m["policy_final_states"].append(pk)
                    m["final_states_bytes"].append(ri["feats"].numel() * ri["feats"].element_size() / GIB)
                    k = max(1, a.reprobe_claim_tokens // ri["feats"].shape[1])
                    calls = [(ri["claims"][c:c + k], ri["owner"][c:c + k]) for c in range(0, len(ri["owner"]), k)]
                    pk, _ = peak(lambda: [run_head(head, ri["feats"], ri["valid"], c, o, "visible") for c, o in calls])
                    m["reprobe_head"].append(pk)
                    del ri, calls
                    torch.cuda.empty_cache()
            res[f"b{B}"] = {"groups": len(groups), "resident": resident, **{k: summary(v) for k, v in m.items()}}
            print(f"[b{B}] {len(groups)} calls, resident {resident:.2f} GiB: "
                  + ", ".join(f"{k} max {summary(v)['max']:.3f} mean {summary(v)['mean']:.3f}" for k, v in m.items()),
                  flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"[out] {a.out}", flush=True)


if __name__ == "__main__":
    main()
