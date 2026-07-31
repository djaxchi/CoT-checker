#!/usr/bin/env python3
"""Encode the future-delta (pcd) representation for PRM800K, from generated futures.

Consumes the augmented split (generate_prm800k_next_step.py output: each row has a
`generated_next_step`). For each step it runs ONE causal pass over
    "Problem:\\n{problem}\\n\\nSolution:\\n{prior}\\n\\n{current}\\n\\n{next}"
and builds the vector the ceiling test picked as the winner:
    pcd = concat[ past boundary state, mean(current step tokens),
                  within-step delta of the NEXT step (S_next^end - S_next^preboundary) ]
The future enters only as a transition (delta), which stays localized to the onset
instead of leaking a persistent "error region" level (see the ceiling result).

Writes the dense harness contract {stem}_h.shard{i}.npy + {stem}_y.shard{i}.npy
(order-invariant for a PRM linear probe; the slurm concatenates shards). Sharded
whole-node H100.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def tokenize_pcd(tok, problem, prefix, current, nxt):
    """Return (ids, cur_start, cur_len, next_start, next_len) with exact boundaries."""
    prior = prefix.strip()
    pre_text = f"Problem:\n{problem}\n\nSolution:\n" + (prior + "\n\n" if prior else "")
    ids = list(tok(pre_text, add_special_tokens=True)["input_ids"])
    sep = tok("\n\n", add_special_tokens=False)["input_ids"]
    cur_start = len(ids)
    cur_ids = tok(current.strip(), add_special_tokens=False)["input_ids"]
    ids += cur_ids
    ids += sep
    next_start = len(ids)
    next_ids = tok(nxt.strip(), add_special_tokens=False)["input_ids"] if nxt.strip() else []
    ids += next_ids
    return ids, cur_start, len(cur_ids), next_start, len(next_ids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in_jsonl", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--stem", required=True)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only, torch_dtype=torch.float16,
    ).to(device).eval()
    d = model.config.hidden_size

    rows = [json.loads(l) for l in args.in_jsonl.read_text().splitlines() if l.strip()]
    for gi, r in enumerate(rows):
        r["_gi"] = gi
    rows = [r for r in rows if r["_gi"] % args.num_shards == args.shard_idx]

    toks, keep = [], []
    n_skip = 0
    for r in rows:
        ids, cs, cl, ns, nl = tokenize_pcd(tok, r["problem"], r["prefix"],
                                           r["candidate_step"], r.get("generated_next_step", ""))
        if cl == 0 or len(ids) > args.max_seq_len:
            n_skip += 1
            continue
        toks.append((ids, cs, cl, ns, nl)); keep.append(r)

    out = np.empty((len(toks), 3 * d), dtype=np.float16)
    y = np.array([int(r["label"]) for r in keep], dtype=np.int8)
    t0 = time.perf_counter()
    for i in range(0, len(toks), args.batch_size):
        batch = toks[i:i + args.batch_size]
        maxlen = max(len(t[0]) for t in batch)
        padded = [t[0] + [pad_id] * (maxlen - len(t[0])) for t in batch]
        mask = [[1] * len(t[0]) + [0] * (maxlen - len(t[0])) for t in batch]
        inp = torch.tensor(padded, dtype=torch.long, device=device)
        att = torch.tensor(mask, dtype=torch.long, device=device)
        with torch.no_grad():
            o = model(inp, attention_mask=att, output_hidden_states=True, use_cache=False)
        hs = o.hidden_states[args.layer]; del o
        for b, (ids, cs, cl, ns, nl) in enumerate(batch):
            h = hs[b].float()
            past = h[cs - 1]
            cur = h[cs:cs + cl].mean(0)
            if nl > 0:
                delta = h[ns + nl - 1] - h[ns - 1]        # within-step delta of next
            else:
                delta = torch.zeros(d, device=h.device)
            out[i + b] = torch.cat([past, cur, delta]).to(torch.float16).cpu().numpy()
        del hs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / f"{args.stem}_h.shard{args.shard_idx}.npy", out)
    np.save(args.out_dir / f"{args.stem}_y.shard{args.shard_idx}.npy", y)
    print(f"[pcd:{args.stem}:s{args.shard_idx}] {out.shape} kept={len(toks)} skip={n_skip} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
