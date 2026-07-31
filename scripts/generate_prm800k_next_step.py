#!/usr/bin/env python3
"""Generate the NEXT reasoning step for each PRM800K training step (horizon W=1).

The future-delta (pcd) representation needs, for a step k, the state trajectory of
the step that FOLLOWS it. PRM800K is a tree of rated candidate steps, so most rows
have no materialized continuation. Here we roll one out with the SAME backbone we
probe (Qwen2.5-7B base): condition on the solution so far (problem + prior steps +
current step) and let the model produce the next step. ProcessBench already ships
real next steps, so only the PRM800K TRAIN side needs generation; the generated
step is added to each row as `generated_next_step`, ready for the pcd encoder.

Split-sharded, whole-node H100. Pure decoding, no hidden states here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def solution_so_far(prefix: str, candidate_step: str) -> str:
    """Prior steps + current step, in the '\\n\\n'-joined solution style."""
    parts = [p for p in (prefix.strip(), candidate_step.strip()) if p]
    return "\n\n".join(parts)


def build_gen_prompt(problem: str, prefix: str, candidate_step: str) -> str:
    """Continuation prompt: the model should emit the next step after this."""
    return f"Problem:\n{problem}\n\nSolution:\n{solution_so_far(prefix, candidate_step)}\n\n"


def extract_next_step(generated_tail: str) -> str:
    """Keep only the first generated step: text up to the first blank-line break."""
    chunk = generated_tail.split("\n\n", 1)[0].strip()
    return chunk


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in_jsonl", required=True, type=Path)
    p.add_argument("--out_jsonl", required=True, type=Path)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in args.in_jsonl.read_text().splitlines() if l.strip()]
    for gi, r in enumerate(rows):
        r["_gi"] = gi
    rows = [r for r in rows if r["_gi"] % args.num_shards == args.shard_idx]
    if args.limit is not None:
        rows = rows[:args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only, torch_dtype=torch.float16,
    ).to(device).eval()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_empty = 0
    with args.out_jsonl.open("w") as f:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            prompts = [build_gen_prompt(r["problem"], r["prefix"], r["candidate_step"]) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1792).to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for r, g in zip(batch, texts):
                nxt = extract_next_step(g)
                n_empty += int(not nxt)
                r.pop("_gi", None)
                r["generated_next_step"] = nxt
                f.write(json.dumps(r) + "\n")
            if (i // args.batch_size) % 20 == 0:
                print(f"[gen] {i+len(batch)}/{len(rows)} empty={n_empty} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"[gen] done {len(rows)} rows, {n_empty} empty next-steps "
          f"({time.perf_counter()-t0:.0f}s) -> {args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
