#!/usr/bin/env python3
"""Stage 1: generate the model's OWN reasoning steps and label them by outcome.

The matched forks use human-written negative steps, so even though Stage 0 ruled out
model surprise, the probe could in principle still read "this step is from the wrong
(human) distribution". The decisive control is to test the probe on the model's *own*
generations, which are on-policy and therefore uniformly low-perplexity by
construction. If the probe still separates steps of correct vs incorrect trajectories,
it is reading correctness, not distribution / surprise.

This script, for a sample of PRM800K problems (taken from the same fork set already
encoded), samples N full solutions per problem, grades each by final-answer match
against ``ground_truth_answer`` (src/eval/math_grade.py), splits each solution into
steps on blank lines (the PRM800K "\\n\\n" step convention) and emits one *item* per
step in the exact schema ``encode_prm800k_forks.py`` consumes, carrying the
trajectory's correctness as the step label.

Outputs (``{stem}`` gains a ``.shardNN`` suffix when --num_shards > 1; generation
is single-device, so a whole H100 node runs one shard per GPU under
CUDA_VISIBLE_DEVICES and scripts/onpolicy/build_pb_traces.py merges them):
  {stem}_items.jsonl         per-step items (problem, prefix, candidate_step, label,
                             role="generated", item_uid, fork_id, traj_correct, ...)
                             -> feed to encode_prm800k_forks.py + encode_fork_confidence.py
  {stem}_trajectories.jsonl  one row per generated solution (text, pred, gold, correct)
  {stem}_generation_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl, write_jsonl  # noqa: E402
from src.eval.math_grade import grade  # noqa: E402
from src.onpolicy.prompts import generation_prompt  # noqa: E402

_BLANKLINE = re.compile(r"\n\s*\n")


def split_into_steps(solution: str) -> list[str]:
    """Split a generated solution into steps on blank lines (PRM800K \\n\\n convention).

    Falls back to single-newline splitting if the model emitted no blank lines but the
    text is clearly multi-line; always strips empties.
    """
    sol = solution.strip()
    if not sol:
        return []
    parts = [p.strip() for p in _BLANKLINE.split(sol) if p.strip()]
    if len(parts) <= 1 and sol.count("\n") >= 2:
        parts = [p.strip() for p in sol.split("\n") if p.strip()]
    return parts


def unique_problems(fork_items: list[dict], id_field: str = "fork_id") -> list[dict]:
    """One (problem, ground_truth_answer) per group id, gold answer present.

    `id_field` lets this read any per-step jsonl, not only the S3 fork set: the
    PRM800K splits group by `problem_id`, which is what the on-policy arm draws
    from so its problems are the ones the off-policy grid was scored on.
    """
    seen: dict[str, dict] = {}
    for it in fork_items:
        fid = it.get(id_field)
        gt = (it.get("ground_truth_answer") or "").strip()
        if fid is None or not gt or fid in seen:
            continue
        seen[fid] = {"fork_id": str(fid), "problem": it["problem"],
                     "ground_truth_answer": gt}
    return list(seen.values())


def build_step_items(problem: str, gold: str, solution: str, traj_uid: str,
                     traj_correct: bool) -> list[dict]:
    """Turn one graded trajectory into per-step encodable items (outcome-labelled)."""
    steps = split_into_steps(solution)
    items = []
    for k, step in enumerate(steps):
        prefix = "\n\n".join(steps[:k])
        items.append({
            "item_uid": f"{traj_uid}::step{k}",
            "fork_id": traj_uid,                 # group steps of one trajectory
            "role": "generated",
            "problem": problem,
            "ground_truth_answer": gold,
            "prefix": prefix,
            "candidate_step": step,
            "label": 0 if traj_correct else 1,   # outcome label inherited by each step
            "traj_correct": bool(traj_correct),
            "step_idx": k,
            "n_steps": len(steps),
        })
    return items


def shard_problems(problems: list[dict], shard_idx: int, num_shards: int) -> list[dict]:
    """The shard's slice of the problem list.

    Striding rather than blocking so every shard sees the same mix of easy and
    hard problems and finishes at roughly the same time; blocks would leave one
    GPU running long after the others. The slices partition the list exactly,
    which is what lets the merge refuse duplicate trajectory ids.
    """
    if not 0 <= shard_idx < num_shards:
        raise ValueError(f"shard_idx {shard_idx} outside 0..{num_shards - 1}")
    return problems[shard_idx::num_shards]


def build_prompt(problem: str) -> str:
    """The sampling prompt. Defined in src/onpolicy/prompts.py so the encoder can
    rebuild the exact context this sampler ran under (--prompt_style generation)."""
    return generation_prompt(problem)


def generate_solutions(problems, tokenizer, model, device, args) -> tuple[list, list]:
    """Returns (step_items, trajectories)."""
    import torch

    step_items: list[dict] = []
    trajectories: list[dict] = []
    t0 = time.perf_counter()
    n = len(problems)
    for pi, prob in enumerate(problems):
        prompt = build_prompt(prob["problem"])
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[1]
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                          num_return_sequences=args.n_samples,
                          pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if args.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=args.temperature,
                              top_p=args.top_p)
            if args.top_k > 0:
                gen_kwargs["top_k"] = args.top_k
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        for s in range(out.shape[0]):
            text = tokenizer.decode(out[s, prompt_len:], skip_special_tokens=True)
            g = grade(text, prob["ground_truth_answer"])
            traj_uid = f"onpolicy::{prob['fork_id']}::g{s}"
            trajectories.append({
                "traj_uid": traj_uid, "fork_id": prob["fork_id"],
                "problem": prob["problem"], "gold": prob["ground_truth_answer"],
                "pred": g["pred"], "correct": g["correct"],
                "gradeable": g["gradeable"], "solution": text,
            })
            if not g["gradeable"]:
                continue                          # cannot label -> drop from probe test
            step_items.extend(build_step_items(
                prob["problem"], prob["ground_truth_answer"], text, traj_uid,
                g["correct"]))
        if (pi + 1) % 20 == 0 or pi + 1 == n:
            done = sum(t["correct"] for t in trajectories)
            print(f"[gen] {pi+1}/{n} problems  ({time.perf_counter()-t0:.0f}s)  "
                  f"correct-so-far={done}/{len(trajectories)}", flush=True)
    return step_items, trajectories


def main() -> None:
    p = argparse.ArgumentParser(description="Generate + grade on-policy reasoning steps.")
    p.add_argument("--fork_items", type=Path, required=True,
                   help="Per-step jsonl to draw problems from (S3 forks, or a "
                        "PRM800K split with --id_field problem_id).")
    p.add_argument("--id_field", type=str, default="fork_id",
                   help="Field that groups rows into problems.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--stem", type=str, default="onpolicy_val")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=4, help="samples per problem")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_problems", type=int, default=300,
                   help="Cap applied to the FULL problem list, before sharding, so "
                        "every shard of a run covers the same problem set.")
    p.add_argument("--top_k", type=int, default=50,
                   help="ReProbe's setting is 50. 0 leaves the model's own "
                        "generation_config value in place.")
    p.add_argument("--shard_idx", type=int, default=0,
                   help="Generation is single-device; a whole H100 node runs four "
                        "shards under CUDA_VISIBLE_DEVICES, which is where the 4x "
                        "comes from. Output files carry the shard suffix.")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_dtype", choices=["float16", "bfloat16", "float32"],
                   default="float16",
                   help="Use the backbone's training dtype; Qwen3 ships bfloat16.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    import torch

    if not 0 <= args.shard_idx < args.num_shards:
        sys.exit(f"[gen] shard_idx {args.shard_idx} outside 0..{args.num_shards-1}")
    suffix = "" if args.num_shards == 1 else f".shard{args.shard_idx:02d}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items_path = args.out_dir / f"{args.stem}{suffix}_items.jsonl"
    if items_path.exists() and not args.force:
        sys.exit(f"[gen] Refusing to overwrite {items_path}. Pass --force.")

    torch.manual_seed(args.seed)
    all_problems = unique_problems(read_jsonl(args.fork_items), args.id_field)
    if args.max_problems > 0:
        all_problems = all_problems[:args.max_problems]
    problems = shard_problems(all_problems, args.shard_idx, args.num_shards)
    print(f"[gen] shard {args.shard_idx}/{args.num_shards}: {len(problems)} of "
          f"{len(all_problems)} problems x {args.n_samples} samples "
          f"(T={args.temperature}, top_p={args.top_p}, top_k={args.top_k})", flush=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            dtype=dtype_map[args.model_dtype])
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            torch_dtype=dtype_map[args.model_dtype])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    step_items, trajectories = generate_solutions(problems, tokenizer, model, device, args)

    write_jsonl(items_path, step_items)
    write_jsonl(args.out_dir / f"{args.stem}{suffix}_trajectories.jsonl", trajectories)
    n_corr = sum(t["correct"] for t in trajectories)
    n_grad = sum(t["gradeable"] for t in trajectories)
    manifest = {
        "run_name": args.run_name, "model": args.model_name_or_path,
        "n_problems": len(problems), "n_problems_all_shards": len(all_problems),
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
        "n_samples": args.n_samples, "prompt_style": "generation",
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "n_trajectories": len(trajectories), "n_gradeable": n_grad,
        "n_correct": n_corr, "n_incorrect": n_grad - n_corr,
        "n_step_items": len(step_items),
        "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": git_commit(),
    }
    (args.out_dir / f"{args.stem}{suffix}_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[gen] trajectories={len(trajectories)} gradeable={n_grad} "
          f"correct={n_corr} incorrect={n_grad-n_corr}  steps={len(step_items)}")
    print(f"[gen] wrote {items_path}")


if __name__ == "__main__":
    main()
