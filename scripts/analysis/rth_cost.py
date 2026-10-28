#!/usr/bin/env python3
"""Wall time to score one trajectory: RTH against a passive probe and the PRM (Stage E1).

On the same GPU, for the same N trajectories, one trajectory at a time (a verifier
scoring candidates as they arrive), CUDA-synchronised:

  rth       every step read under the verifier template plus the read token, the
            trajectory's steps packed into left-padded batches of at most
            --max_tokens padded tokens, as in score_traces_with_read_token.py (each
            step needs its own prefix pass)
  rth_branched  the same reads sharing the trajectory's common prefixes
            (read_token_probe.read_branched, `--branched` in the scorer)
  passive   one Qwen3-8B forward over the whole trajectory, the pass a probe on the
            policy's states reads (the probe head itself costs microseconds)
  prm       one Qwen2.5-Math-PRM-7B forward over its native template with an
            <extra_0> separator after every step (the head's cost is negligible)
  rth_inline    an RTH-I token (--inline_ckpt, scripts/read_token_inline.py): the
            trajectory once in the chat layout, then its read tokens in one call
  inline_margin the read call alone, over a cache the text pass already built: what
            RTH-I adds to a generation that keeps its cache

--methods picks a subset (default: all that the given checkpoints allow).

Reports seconds per trajectory and per step, and tokens processed.

    python scripts/analysis/rth_cost.py --ckpt tokens/dq_correct/read_token.pt \
        --trajectories <pool>/tts_math500.shard00_trajectories.jsonl --n 200 --out e1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from scripts.onpolicy.score_traces_with_read_token import read_jsonl, step_items  # noqa: E402
from scripts.read_token_probe import (  # noqa: E402
    batches, common_prefix, read_all_states, read_branched, tokenize)
from scripts.read_token_store import load_model, load_read_token  # noqa: E402
from src.onpolicy.prompts import verifier_prefix  # noqa: E402

METHODS = ("rth", "rth_branched", "passive", "rth_inline", "inline_margin", "prm")

PRM_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def timed(dev, f) -> float:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        f()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t


def summary(sec: list[float], steps: list[int], tokens: list[int]) -> dict:
    return {"traces": len(sec), "sec_per_trace": float(np.mean(sec)),
            "sec_per_step": float(np.sum(sec) / np.sum(steps)),
            "tokens_per_trace": float(np.mean(tokens)), "total_sec": float(np.sum(sec))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--prm_model", default="Qwen/Qwen2.5-Math-PRM-7B")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--inline_ckpt", type=Path, default=None, help="an RTH-I token, for the inline rows")
    p.add_argument("--methods", default=None, help=f"comma list from {','.join(METHODS)}")
    p.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max_tokens", type=int, default=16000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = [r for path in a.trajectories for r in read_jsonl(path)
            if r.get("gradeable") and (r.get("solution") or "").strip()][:a.n + a.warmup]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res: dict = {"n": a.n, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}

    model, tok = load_model(a.model, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rt, ck = load_read_token(a.ckpt, model.config.hidden_size, dev, model)
    methods = (a.methods.split(",") if a.methods else
               [m for m in METHODS if a.inline_ckpt is not None or m not in ("rth_inline", "inline_margin")])
    if set(methods) - set(METHODS):
        raise SystemExit(f"unknown methods {set(methods) - set(METHODS)}")
    if a.inline_ckpt is not None:
        from scripts.read_token_inline import build_sequences, marker_from_ckpt, read_pass, row, text_pass
        irt, ick = load_read_token(a.inline_ckpt, model.config.hidden_size, dev, model)
        imk = marker_from_ckpt(ick, dev)
    elif {"rth_inline", "inline_margin"} & set(methods):
        raise SystemExit("the inline rows need --inline_ckpt")
    for name in [m for m in methods if m != "prm"]:
        sec, steps, tokens = [], [], []
        for i, r in enumerate(rows):
            items = step_items(r)
            if name in ("rth_inline", "inline_margin"):
                ich = bool(ick.get("chain"))
                seqs = [row(q, imk) for q in build_sequences(tok, items, "chat", chain=ich)[0]]
                sa, per = ick.get("soft_attn"), 1 if imk is None else 2
                mark = imk is not None
                if name == "rth_inline":
                    t = timed(dev, lambda: read_pass(model, irt, *text_pass(model, seqs, pad, dev, marker=mark, chain=ich),
                                                     grad=False, soft_attn=sa, mk=imk))
                    n_tok = sum(len(q[0]) + per * len(q[1]) for q in seqs)
                else:
                    with torch.no_grad():
                        cache, lay = text_pass(model, seqs, pad, dev, marker=mark, chain=ich)
                    t = timed(dev, lambda: read_pass(model, irt, cache, lay, grad=False, soft_attn=sa, mk=imk))
                    n_tok = sum(per * len(q[1]) for q in seqs)
            elif name == "rth":
                ids = [tokenize(tok, it) for it in items]
                t = timed(dev, lambda: [read_all_states(model, rt, [ids[j] for j in b], pad, dev,
                                                        grad=False, soft_attn=ck.get("soft_attn"))
                                        for b in batches([len(x) for x in ids], a.max_tokens, None)])
                n_tok = sum(len(x) + rt.m for x in ids)
            elif name == "rth_branched":
                ids = [tokenize(tok, it) for it in items]
                t = timed(dev, lambda: read_branched(model, rt, ids, pad, dev,
                                                     soft_attn=ck.get("soft_attn")))
                trunk = max(ids, key=len)
                n_tok = len(trunk) + sum(len(x) - common_prefix(x, trunk) + rt.m for x in ids)
            else:
                text = verifier_prefix(r["problem"], "") + "\n\n".join(s["step"] for s in items)
                ids = torch.tensor([tok(text)["input_ids"]], device=dev)
                t = timed(dev, lambda: model.model(input_ids=ids))
                n_tok = ids.shape[1]
            if i >= a.warmup:
                sec.append(t)
                steps.append(len(items))
                tokens.append(n_tok)
        res[name] = summary(sec, steps, tokens)
        print(f"[e1] {name}: {res[name]}", flush=True)
    del model
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    if "prm" not in methods:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        return

    pm, ptok = load_model(a.prm_model, dev)
    sec, steps, tokens = [], [], []
    for i, r in enumerate(rows):
        st = split_into_steps(r["solution"])
        text = ptok.apply_chat_template(
            [{"role": "system", "content": PRM_SYSTEM}, {"role": "user", "content": r["problem"]},
             {"role": "assistant", "content": "<extra_0>".join(st) + "<extra_0>"}],
            tokenize=False, add_generation_prompt=False)
        ids = torch.tensor([ptok(text)["input_ids"]], device=dev)
        t = timed(dev, lambda: pm.model(input_ids=ids))
        if i >= a.warmup:
            sec.append(t)
            steps.append(len(st))
            tokens.append(ids.shape[1])
    res["prm"] = summary(sec, steps, tokens)
    print(f"[e1] prm: {res['prm']}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
