#!/usr/bin/env python3
"""Recover per-token logprobs for already-generated trajectories.

REPORT.md §20.11 recorded that token confidence "cannot be computed at all from
what is on disk, because the saved trajectories carry no logprobs", and §20.14
named that comparison "the single most valuable thing to regenerate". The
premise is true of the saved *files* and false of the saved *data*: the
trajectory text is on disk and `src/onpolicy/prompts.py` rebuilds the exact
prompt, so one teacher-forced forward pass per trajectory recovers the model's
distribution at every generated position. No regeneration is needed, and the
go/no-go gate of docs/onpolicy_tiebreak_v2_plan.md Phase 1 runs on the existing
2,873-trajectory pool in minutes rather than on a fresh 64K-trajectory run.

What recomputation does and does not reproduce. The logprobs here are the
model's raw next-token distribution, which is what DeepConf's Eq 2 reads, and
they are *not* the temperature/top-p/top-k-modified distribution the sampler
drew from. That is the correct quantity, not an approximation of it. The only
discrepancy against generation time is bf16 kernel nondeterminism, which
§20.10 measured at up to 0.0064 on a probe score when the same step is run
alone versus in a batch, and which is negligible against confidence values of
order 1 to 10 nats.

Outputs, per shard:
  {stem}.shardNN_conf.npz    per-trajectory token confidence, sampled logprob,
                             top1/top2 logprob, step spans, answer span
  {stem}.shardNN_conf.jsonl  one row per trajectory: every Tier-1 rule's scalar
  {stem}.shardNN_conf_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl, write_jsonl  # noqa: E402
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from src.analysis import token_confidence as tc  # noqa: E402
from src.onpolicy.prompts import generation_prompt  # noqa: E402
from src.onpolicy.spans import (answer_char_span, char_span_to_token_span,  # noqa: E402
                                step_token_spans, verify_spans_cover)


def shard(rows: list, idx: int, n: int) -> list:
    if not 0 <= idx < n:
        raise ValueError(f"shard_idx {idx} outside 0..{n - 1}")
    return rows[idx::n]


def main() -> None:
    p = argparse.ArgumentParser(description="Recover token logprobs for saved trajectories.")
    p.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--stem", type=str, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--model_dtype", choices=["float16", "bfloat16", "float32"],
                   default="bfloat16")
    p.add_argument("--topk", type=int, default=20,
                   help="k in DeepConf Eq 2. Their Qwen3-8B runs sample at "
                        "top_k=20; the confidence k is a separate choice and is "
                        "recorded in the manifest.")
    p.add_argument("--max_traces", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    suffix = "" if args.num_shards == 1 else f".shard{args.shard_idx:02d}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = args.out_dir / f"{args.stem}{suffix}_conf.npz"
    if out_npz.exists() and not args.force:
        sys.exit(f"[conf] Refusing to overwrite {out_npz}. Pass --force.")

    rows: list[dict] = []
    for path in args.trajectories:
        rows.extend(read_jsonl(path))
    rows = [r for r in rows if r.get("gradeable") and (r.get("solution") or "").strip()]
    if args.max_traces > 0:
        rows = rows[:args.max_traces]
    mine = shard(rows, args.shard_idx, args.num_shards)
    print(f"[conf] shard {args.shard_idx}/{args.num_shards}: {len(mine)} of "
          f"{len(rows)} gradeable trajectories, k={args.topk}", flush=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.model_dtype]
    tokzr = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                          local_files_only=args.local_files_only)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    def n_tokens(s: str) -> int:
        return len(tokzr(s, add_special_tokens=False)["input_ids"])

    rules = tc.trace_rules()
    store: dict[str, np.ndarray] = {}
    scalars: list[dict] = []
    span_ok = span_total = 0
    t0 = time.perf_counter()

    for i, row in enumerate(mine):
        prompt = generation_prompt(row["problem"])
        solution = row["solution"]
        p_ids = tokzr(prompt, add_special_tokens=False)["input_ids"]
        enc = tokzr(solution, add_special_tokens=False, return_offsets_mapping=True)
        s_ids, offsets = enc["input_ids"], enc["offset_mapping"]
        if not s_ids:
            continue
        ids = torch.tensor([p_ids + s_ids], device=device)
        with torch.no_grad():
            logits = model(ids).logits[0]
        # Position t predicts token t+1, so the distribution for generated token
        # j sits at index len(prompt)+j-1.
        start = len(p_ids) - 1
        gen_logits = logits[start:start + len(s_ids)].float()
        lp = torch.log_softmax(gen_logits, dim=-1)
        top = lp.topk(args.topk, dim=-1).values.cpu().numpy()
        sampled = lp.gather(1, torch.tensor(s_ids, device=device).unsqueeze(1))
        sampled = sampled.squeeze(1).cpu().numpy()
        del logits, gen_logits, lp

        conf = tc.token_confidence(top)
        steps = split_into_steps(solution)
        spans = step_token_spans(prompt, steps, n_tokens)
        spans = [(a, min(b, len(s_ids))) for a, b in spans]
        span_total += 1
        span_ok += int(verify_spans_cover(spans, len(s_ids), tol=3))
        ch = answer_char_span(solution)
        ans = char_span_to_token_span(offsets, ch) if ch else (0, 0)

        uid = row["traj_uid"]
        store[f"{uid}::conf"] = conf.astype(np.float32)
        store[f"{uid}::sampled_lp"] = sampled.astype(np.float32)
        store[f"{uid}::top2"] = top[:, :2].astype(np.float32)
        store[f"{uid}::spans"] = np.asarray(spans, dtype=np.int32)
        store[f"{uid}::answer_span"] = np.asarray(ans, dtype=np.int32)

        vals = {name: float(fn(conf, spans, top)) for name, fn in rules.items()}
        vals["mean_sampled_logprob"] = float(sampled.mean())
        vals["min_sampled_logprob"] = float(sampled.min())
        vals["answer_token_margin"] = float(tc.answer_token_margin(top, ans))
        scalars.append({"traj_uid": uid, "fork_id": row.get("fork_id"),
                        "correct": bool(row["correct"]), "n_gen_tokens": len(s_ids),
                        "n_steps": len(steps), "rules": vals})

        if (i + 1) % 200 == 0 or i + 1 == len(mine):
            print(f"[conf] {i+1}/{len(mine)}  ({time.perf_counter()-t0:.0f}s)  "
                  f"span_ok={span_ok}/{span_total}", flush=True)

    np.savez_compressed(out_npz, **store)
    write_jsonl(args.out_dir / f"{args.stem}{suffix}_conf.jsonl", scalars)
    manifest = {
        "stem": args.stem, "model": args.model_name_or_path, "topk": args.topk,
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
        "n_traces": len(scalars), "rules": sorted(rules) + [
            "mean_sampled_logprob", "min_sampled_logprob", "answer_token_margin"],
        "span_coverage_pass": span_ok, "span_coverage_total": span_total,
        "span_coverage_rate": (span_ok / span_total) if span_total else None,
        "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": git_commit(),
    }
    (args.out_dir / f"{args.stem}{suffix}_conf_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[conf] wrote {out_npz}  traces={len(scalars)}  "
          f"span_coverage={span_ok}/{span_total}")
    # A low coverage rate means the step spans are misaligned and every per-step
    # confidence built on them is wrong. Fail loudly rather than write a number.
    if span_total and span_ok / span_total < 0.90:
        sys.exit(f"[conf] FATAL span coverage {span_ok}/{span_total} below 0.90")


if __name__ == "__main__":
    main()
