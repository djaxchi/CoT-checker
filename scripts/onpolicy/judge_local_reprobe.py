#!/usr/bin/env python3
"""Annotate on-policy trajectories with a locally hosted judge, ReProbe protocol.

Compute nodes have no network, so the primary annotator runs from a local
snapshot rather than an API. The protocol is the paper's and is documented in
docs/reprobe_label_semantics.md: the judge sees the problem, the ground-truth
answer and the numbered steps, and reports the SET of faulty steps. Steps after
a faulty one are not marked faulty, because the paper does not do that.

Everything about a run is recoverable from its output. Each row carries the raw
judge text, the parsed faulty set, the derived per-step binary labels, the
first-error index for the evaluation code that wants one, both model
identities, the prompt version, the git commit and the seed. A run appends as it
goes and skips trajectories already present, so a walltime kill costs the
trajectory in flight and nothing else.

Sharded in-node over CUDA_VISIBLE_DEVICES the way every other job here is, or
run as one process with device_map="auto" when the model needs several GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl  # noqa: E402
from scripts.onpolicy.judge_steps import (  # noqa: E402
    build_prompt_reprobe, parse_step_set, step_labels_from_faulty,
)

PROMPT_VERSION = "reprobe-faulty-set-v1"


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["traj_uid"] for r in read_jsonl(path)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, type=Path,
                   help="jsonl with id, problem, steps, gold, traj_correct")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model_path", required=True,
                   help="Local snapshot directory. Never a hub id on a compute node.")
    p.add_argument("--generator", default="Qwen/Qwen3-8B-Base",
                   help="Recorded with every row so the pool is identifiable later.")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=320)
    p.add_argument("--max_prompt_tokens", type=int, default=3072)
    p.add_argument("--max_traces", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert os.path.isdir(args.model_path), f"not a local dir: {args.model_path}"
    torch.manual_seed(args.seed)

    traces = read_jsonl(args.traces)
    if args.max_traces > 0:
        traces = traces[:args.max_traces]
    traces = traces[args.shard_idx::args.num_shards]
    done = load_done(args.out)
    todo = [t for t in traces if t["id"] not in done]
    print(f"[judge] shard {args.shard_idx}/{args.num_shards}: {len(todo)} to do "
          f"({len(done)} already annotated)", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, dtype=args.dtype,
        device_map=args.device_map)
    model.eval()
    print(f"[judge] model loaded in {time.perf_counter()-t_load:.0f}s", flush=True)
    device = next(model.parameters()).device

    args.out.parent.mkdir(parents=True, exist_ok=True)
    chat = getattr(tok, "chat_template", None) is not None
    n_parsed = n_fail = 0
    t0 = time.perf_counter()
    with args.out.open("a") as fh:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            prompts = [build_prompt_reprobe(t["problem"], t["steps"],
                                            t.get("gold") or "", tok, chat)
                       for t in batch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_prompt_tokens).to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            width = enc["input_ids"].shape[1]
            texts = tok.batch_decode(out[:, width:], skip_special_tokens=True)
            for t, raw in zip(batch, texts):
                faulty = parse_step_set(raw, len(t["steps"]))
                ok = faulty is not None
                n_parsed += int(ok)
                n_fail += int(not ok)
                labels = step_labels_from_faulty(faulty or [], len(t["steps"]))
                fh.write(json.dumps({
                    "traj_uid": t["id"], "id": t["id"],
                    "problem_id": t.get("problem_id"),
                    "gold": t.get("gold"),
                    "traj_correct": t.get("traj_correct"),
                    "n_steps": len(t["steps"]),
                    "faulty_steps": faulty,
                    "step_labels": labels if ok else None,
                    # kept so the evaluation code that wants a single index has
                    # one, derived rather than asked for: the paper's protocol
                    # reports a set and the earliest member is its first error
                    "first_error": (min(faulty) if faulty else -1) if ok else -1,
                    "parse_ok": ok,
                    "raw": raw[:2000],
                    "prompt_version": PROMPT_VERSION,
                    "judge_model": args.model_path,
                    "generator": args.generator,
                    "seed": args.seed,
                    "code_commit": git_commit(),
                }) + "\n")
            fh.flush()
            n = i + len(batch)
            if (i // args.batch_size) % 5 == 0 or n >= len(todo):
                el = time.perf_counter() - t0
                print(f"[judge] {n}/{len(todo)} ({el:.0f}s, {n/max(el,1e-9)*3600:.0f}/h) "
                      f"parsed {n_parsed} failed {n_fail}", flush=True)

    rep = {"n_done": n_parsed + n_fail, "n_parsed": n_parsed, "n_parse_fail": n_fail,
           "parse_failure_rate": n_fail / max(1, n_parsed + n_fail),
           "seconds": round(time.perf_counter() - t0, 1),
           "throughput_per_hour": round((n_parsed + n_fail) /
                                        max(time.perf_counter() - t0, 1e-9) * 3600, 1),
           "model_path": args.model_path, "prompt_version": PROMPT_VERSION,
           "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
           "shard_idx": args.shard_idx, "num_shards": args.num_shards,
           "created_at": datetime.now(timezone.utc).isoformat(),
           "code_commit": git_commit()}
    print(json.dumps(rep, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
