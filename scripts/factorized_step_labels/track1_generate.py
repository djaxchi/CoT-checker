#!/usr/bin/env python3
"""Track 1 of factorized_step_labels_v1: downstream risk by free generation.

Plan: docs/factorized_step_labels_v1_plan.md, section 2.

For every fork of the frozen split (this shard's share):
  A    prompt + steps 1..i (the real step i), k continuations to the end;
  C_j  m alternative steps i sampled from prompt + steps 1..i-1, each stopped at
       its first newline token, then k continuations from each.
Contexts are token ids from the split (ReProbe's released ids under
`reprobe_prompt`), never re-tokenized text; sampled ids are appended as generated.
The reply cap counts from the start of the reply, prefix included, as in the pool.

Grading and risk statistics live in src/analysis/factorized_step_labels; all
continuations of one fork are graded together so one answer identity gets one
grade. Output per shard: track1.shardNN.jsonl (one row per fork) and a manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.factorized_step_labels import grade_fork, risk_stats  # noqa: E402

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def trim_after_eos(ids: list[int], eos_ids: set[int]) -> tuple[list[int], bool]:
    """Generated ids through the first EOS (kept), dropping the padding after it.

    Qwen3's pad id is also one of its EOS ids, so padding cannot be stripped from
    the right without losing a real EOS; the first EOS is the end.
    """
    for j, t in enumerate(ids):
        if t in eos_ids:
            return ids[:j + 1], True
    return ids, False


def cut_at_newline(ids: list[int], decode, eos_ids: set[int]) -> tuple[list[int], str]:
    """Generated step ids up to and including the first token holding a newline.

    Returns (ids, status): "newline" for a normal step line, "eos" when the model
    ended the reply first (an answer line ending in <|im_end|>; the EOS is kept),
    "cap" when neither happened within the step budget.
    """
    for j, t in enumerate(ids):
        if t in eos_ids:
            return ids[:j + 1], "eos"
        if "\n" in decode([t]):
            return ids[:j + 1], "newline"
    return ids, "cap"


def sample(model, ctx: list[int], n: int, max_new: int, args, device, pad_id, **kw) -> list[list[int]]:
    import torch
    x = torch.tensor([ctx], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True,
                             temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                             max_new_tokens=max_new, num_return_sequences=n,
                             pad_token_id=pad_id, **kw)
    return [out[r, len(ctx):].tolist() for r in range(n)]


def continuations(model, tok, fork, ctx_reply: list[int], args, device, pad_id, eos_ids):
    """k continuations after prompt + ctx_reply, as whole-reply records."""
    budget = args.max_reply_tokens - len(ctx_reply)
    if budget <= 0:
        return [{"gen_ids": [], "n_new": 0, "eos": False, "truncated": True,
                 "text": tok.decode(ctx_reply, skip_special_tokens=True)}] * args.k
    gens = sample(model, fork["prompt_ids"] + ctx_reply, args.k, budget, args, device, pad_id)
    out = []
    for g in gens:
        g, eos = trim_after_eos(g, eos_ids)
        out.append({"gen_ids": g, "n_new": len(g), "eos": eos, "truncated": not eos,
                    "text": tok.decode(ctx_reply + g, skip_special_tokens=True)})
    return out


def run_fork(model, tok, fork, fi: int, args, device, pad_id, eos_ids) -> dict:
    import torch
    rep = fork["reply_ids"]
    lo, hi = fork["step_spans"][fork["step"]]
    prefix, real = rep[:lo], rep[lo:hi]
    decode = lambda ids: tok.decode(ids)  # noqa: E731

    torch.manual_seed(args.seed * 100003 + fi)
    # stop_strings only saves time (a stopped row is padded); the cut below decides
    raw = sample(model, fork["prompt_ids"] + prefix, args.m,
                 min(args.max_step_tokens, args.max_reply_tokens - len(prefix)),
                 args, device, pad_id, stop_strings=["\n"], tokenizer=tok)
    alts = []
    for j, g in enumerate(raw):
        ids, status = cut_at_newline(g, decode, eos_ids)
        text = tok.decode(ids)
        alts.append({"j": j, "ids": ids, "text": text, "status": status,
                     "equals_original": ids == real,
                     "is_answer_line": text.lstrip().lower().startswith("<answer>")})

    conds = [("A", prefix + real, None)] + [(f"C{a['j']}", prefix + a["ids"], a) for a in alts]
    records = {}
    for ci, (name, ctx, alt) in enumerate(conds):
        torch.manual_seed(args.seed * 100003 + fi * 101 + ci + 1)
        if alt is not None and alt["status"] != "newline":
            # A terminal or capped alternative: nothing to continue. The reply as it
            # stands is graded once and repeated, so err(C_j) is 0 or 1.
            eos = alt["status"] == "eos"
            rec = {"gen_ids": [], "n_new": 0, "eos": eos, "truncated": not eos,
                   "text": tok.decode(ctx, skip_special_tokens=True)}
            records[name] = [rec] * args.k
        else:
            records[name] = continuations(model, tok, fork, ctx, args, device, pad_id, eos_ids)
    flat = [(n, r) for n, rs in records.items() for r in rs]
    grades = grade_fork([{"text": r["text"], "truncated": r["truncated"]} for _, r in flat], fork["gold"])
    out_conds = {n: [] for n in records}
    for (n, r), g in zip(flat, grades):
        out_conds[n].append({"gen_text": tok.decode(r["gen_ids"], skip_special_tokens=True),
                             "n_new": r["n_new"], "eos": r["eos"], "truncated": r["truncated"],
                             **g})
    err = {n: [not c["correct"] for c in cs] for n, cs in out_conds.items()}
    stats = risk_stats(err["A"], [err[f"C{a['j']}"] for a in alts], seed=fi)
    return {"fork_id": fork["fork_id"], "fork_index": fi, "validity": fork["validity"],
            "position": fork["position"], "step": fork["step"], "n_steps": fork["n_steps"],
            "orig_correct": fork["orig_correct"], "gold": fork["gold"],
            "real_step_text": tok.decode(real),
            "alternatives": [{k: v for k, v in a.items() if k != "ids"} | {"n_tokens": len(a["ids"])}
                             for a in alts],
            "conditions": out_conds, "risk": stats}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fork_split", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--m", type=int, default=4)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_reply_tokens", type=int, default=4096)
    p.add_argument("--max_step_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="smoke: first N forks of this shard")
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true", help="append to an existing shard, skipping its forks")
    a = p.parse_args()

    out = a.out_dir / f"track1.shard{a.shard_idx:02d}.jsonl"
    done: set[str] = set()
    if out.exists() and a.resume:
        # a shard cut by the walltime keeps every fork it flushed; drop a torn last line
        good = []
        for l in out.read_text().splitlines():
            try:
                good.append(json.loads(l))
            except json.JSONDecodeError:
                break
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good))
        done = {r["fork_id"] for r in good}
    elif out.exists() and not a.force:
        sys.exit(f"[t1] refusing to overwrite {out} (pass --resume to continue it)")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(a.fork_split.read_text())
    forks = [(i, f) for i, f in enumerate(doc["forks"]) if i % a.num_shards == a.shard_idx]
    if a.limit:
        forks = forks[:a.limit]
    n_all = len(forks)
    forks = [(i, f) for i, f in forks if f["fork_id"] not in done]
    if done:
        print(f"[t1] resume: {len(done)} forks already written, {len(forks)} of {n_all} left", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path, revision=a.revision,
                                        local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(a.model_name_or_path, revision=a.revision,
                                                 local_files_only=a.local_files_only,
                                                 dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    print(f"[t1] shard {a.shard_idx}/{a.num_shards}: {len(forks)} forks, m={a.m} k={a.k} "
          f"T={a.temperature} top_p={a.top_p} top_k={a.top_k} cap={a.max_reply_tokens}", flush=True)

    t0 = time.perf_counter()
    with open(out, "a" if done else "w", encoding="utf-8") as fh:
        for n, (fi, f) in enumerate(forks):
            row = run_fork(model, tok, f, fi, a, device, pad_id, eos_ids)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
            r = row["risk"]
            print(f"[t1] {n+1}/{len(forks)} {f['fork_id']} {f['validity']}/{f['position']} "
                  f"err_A={r['err_a']:.3f} err_C={['%.3f' % e for e in r['err_c']]} "
                  f"diff={r['risk_diff']:+.3f}+-{r['risk_diff_se']:.3f} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    (a.out_dir / f"track1.shard{a.shard_idx:02d}.manifest.json").write_text(json.dumps({
        "experiment": "factorized_step_labels_v1", "track": 1, "shard_idx": a.shard_idx,
        "num_shards": a.num_shards, "n_forks": n_all, "n_forks_this_run": len(forks), "args": {k: str(v) for k, v in vars(a).items()},
        "fork_split_sha256": __import__("hashlib").sha256(a.fork_split.read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "seconds": time.perf_counter() - t0, "created": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
