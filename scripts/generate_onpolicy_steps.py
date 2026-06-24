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

Outputs:
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


def unique_problems(fork_items: list[dict]) -> list[dict]:
    """One (problem, ground_truth_answer) per fork_id, gold answer present."""
    seen: dict[str, dict] = {}
    for it in fork_items:
        fid = it.get("fork_id")
        gt = (it.get("ground_truth_answer") or "").strip()
        if fid is None or not gt or fid in seen:
            continue
        seen[fid] = {"fork_id": fid, "problem": it["problem"],
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


def build_prompt(problem: str) -> str:
    return (f"Problem:\n{problem}\n\n"
            "Solve the problem step by step. Put each step on its own line, and write "
            "the final answer inside \\boxed{}.\n\nSolution:\n")


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
    p.add_argument("--fork_items", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--stem", type=str, default="onpolicy_val")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=4, help="samples per problem")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_problems", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    import torch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items_path = args.out_dir / f"{args.stem}_items.jsonl"
    if items_path.exists() and not args.force:
        sys.exit(f"[gen] Refusing to overwrite {items_path}. Pass --force.")

    torch.manual_seed(args.seed)
    problems = unique_problems(read_jsonl(args.fork_items))
    if args.max_problems > 0:
        problems = problems[:args.max_problems]
    print(f"[gen] {len(problems)} problems x {args.n_samples} samples "
          f"(T={args.temperature})", flush=True)

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
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
    write_jsonl(args.out_dir / f"{args.stem}_trajectories.jsonl", trajectories)
    n_corr = sum(t["correct"] for t in trajectories)
    n_grad = sum(t["gradeable"] for t in trajectories)
    manifest = {
        "run_name": args.run_name, "model": args.model_name_or_path,
        "n_problems": len(problems), "n_samples": args.n_samples,
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
        "n_trajectories": len(trajectories), "n_gradeable": n_grad,
        "n_correct": n_corr, "n_incorrect": n_grad - n_corr,
        "n_step_items": len(step_items),
        "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": git_commit(),
    }
    (args.out_dir / f"{args.stem}_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[gen] trajectories={len(trajectories)} gradeable={n_grad} "
          f"correct={n_corr} incorrect={n_grad-n_corr}  steps={len(step_items)}")
    print(f"[gen] wrote {items_path}")


if __name__ == "__main__":
    main()
