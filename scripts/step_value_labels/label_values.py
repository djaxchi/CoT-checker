#!/usr/bin/env python3
"""step_value_labels_v1 labeler: k free rollouts from every cut point of a trace.

Plan: docs/step_value_labels_v1_plan.md, section 2.

For every trace of the frozen set (this shard's share) and every cut c in its
`cuts`, k continuations are sampled from prompt + reply[:cut_offsets[c]] to the end
of the reply, under factorized_step_labels_v1's sampler (T=1.0, top-p 0.95,
top-k 50, reply cap 4,096 tokens counted from the reply start). Contexts are the
released token ids, never re-tokenized text.

Throughput: the rollouts of several traces (about --pool_rows rows) are sampled
together in rounds of --chunk new tokens, left-padded batches of at most
--gen_batch rows and --max_batch_tokens padded tokens; rows that have not ended
are re-batched into the next round (src/analysis/step_value_labels.sample_rounds),
so batches stay full and a rare runaway row does not hold a batch to the cap.

All rollouts of a trace are graded together with the project grader (one grade per
answer identity, as in track 1). Output per shard: values.shardNN.jsonl, one row
per trace with per-cut outcomes and per-step risk, plus a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.factorized_step_labels import grade_fork  # noqa: E402
from src.analysis.step_value_labels import sample_rounds, step_risks  # noqa: E402

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def make_gen(model, args, pad_id, device):
    import torch

    def gen(ctxs: list[list[int]], max_new: int) -> list[list[int]]:
        L = max(len(c) for c in ctxs)
        ids = torch.full((len(ctxs), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(ctxs), L), dtype=torch.long)
        for r, c in enumerate(ctxs):
            ids[r, L - len(c):] = torch.tensor(c, dtype=torch.long)
            mask[r, L - len(c):] = 1
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                 max_new_tokens=max_new, pad_token_id=pad_id)
        return [out[r, L:].tolist() for r in range(len(ctxs))]
    return gen


def label_group(gen, tok, traces: list[tuple[int, dict]], args, eos_ids) -> list[dict]:
    """Sample k rollouts per cut for a group of traces at once, grade per trace."""
    rows = []                                     # (trace position, cut, context, budget)
    for ti, (_, t) in enumerate(traces):
        for c, off in zip(t["cuts"], t["cut_offsets"]):
            ctx = t["prompt_ids"] + t["reply_ids"][:off]
            rows += [(ti, c, ctx, args.max_reply_tokens - off)] * args.k
    res = sample_rounds(gen, [r[2] for r in rows], [r[3] for r in rows], eos_ids, args.chunk, args.gen_batch,
                        getattr(args, "max_batch_tokens", None))
    out = []
    for ti, (fi, t) in enumerate(traces):
        mine = [(r, x) for r, x in zip(rows, res) if r[0] == ti]
        replies = []
        for (_, c, _, _), x in mine:
            off = t["cut_offsets"][t["cuts"].index(c)]
            text = tok.decode(t["reply_ids"][:off] + x["ids"], skip_special_tokens=True)
            replies.append({"text": text, "truncated": x["truncated"]})
        grades = grade_fork(replies, t["gold"])
        cuts = {c: [] for c in t["cuts"]}
        for ((_, c, _, _), x), g in zip(mine, grades):
            cuts[c].append({"gen_text": tok.decode(x["ids"], skip_special_tokens=True), "n_new": len(x["ids"]),
                            "eos": x["eos"], "truncated": x["truncated"], "pred": g["pred"],
                            "correct": g["correct"], "gradeable": g["gradeable"]})
        err = {c: [not r["correct"] for r in rs] for c, rs in cuts.items()}
        keep = ("trace_id", "orig_trace_id", "problem_key", "group", "gold", "n_steps", "step_claim_idx",
                "y_ds", "mask_ds", "y_self", "mask_self", "first_error_ds", "orig_correct", "fork_step")
        out.append({**{k: t[k] for k in keep if k in t}, "trace_index": fi,
                    "cuts": [{"cut": c, "offset": off, "k": len(cuts[c]), "err": sum(err[c]) / len(err[c]),
                              "rollouts": cuts[c]} for c, off in zip(t["cuts"], t["cut_offsets"])],
                    "steps": step_risks(err, t["n_steps"])})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace_set", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_reply_tokens", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=256, help="new tokens per sampling round")
    p.add_argument("--gen_batch", type=int, default=192, help="max rows per generate call")
    p.add_argument("--max_batch_tokens", type=int, default=160_000,
                   help="max padded tokens (rows x (longest context + chunk)) per generate call")
    p.add_argument("--pool_rows", type=int, default=2048, help="rollouts sampled together (whole traces)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="smoke: first N traces of this shard")
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true", help="append to an existing shard, skipping its traces")
    a = p.parse_args()

    out = a.out_dir / f"values.shard{a.shard_idx:02d}.jsonl"
    done: set[str] = set()
    if out.exists() and a.resume:
        good = []
        for l in out.read_text().splitlines():
            try:
                good.append(json.loads(l))
            except json.JSONDecodeError:
                break                             # a torn last line from a walltime cut
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good))
        done = {r["trace_id"] for r in good}
    elif out.exists() and not a.force:
        sys.exit(f"[val] refusing to overwrite {out} (pass --resume to continue it)")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(a.trace_set.read_text())
    traces = [(i, t) for i, t in enumerate(doc["traces"]) if i % a.num_shards == a.shard_idx]
    if a.limit:
        traces = traces[:a.limit]
    n_all = len(traces)
    traces = [(i, t) for i, t in traces if t["trace_id"] not in done]
    if done:
        print(f"[val] resume: {len(done)} traces already written, {len(traces)} of {n_all} left", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path, revision=a.revision, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(a.model_name_or_path, revision=a.revision,
                                                 local_files_only=a.local_files_only, dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    gen = make_gen(model, a, pad_id, device)
    print(f"[val] shard {a.shard_idx}/{a.num_shards}: {len(traces)} traces, k={a.k} T={a.temperature} "
          f"top_p={a.top_p} top_k={a.top_k} cap={a.max_reply_tokens} chunk={a.chunk} batch={a.gen_batch} "
          f"max_batch_tokens={a.max_batch_tokens} pool={a.pool_rows}",
          flush=True)

    # pools of whole traces with about pool_rows rollouts each; one seed per pool
    groups, cur, rows = [], [], 0
    for it in traces:
        cur.append(it); rows += len(it[1]["cuts"]) * a.k
        if rows >= a.pool_rows:
            groups.append(cur); cur, rows = [], 0
    if cur:
        groups.append(cur)
    t0, n_done, n_roll = time.perf_counter(), 0, 0
    with open(out, "a" if done else "w", encoding="utf-8") as fh:
        for gi, grp in enumerate(groups):
            torch.manual_seed(a.seed * 100003 + grp[0][0])
            for row in label_group(gen, tok, grp, a, eos_ids):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1; n_roll += sum(c["k"] for c in row["cuts"])
            fh.flush()
            el = time.perf_counter() - t0
            print(f"[val] group {gi+1}/{len(groups)} traces {n_done}/{len(traces)} rollouts {n_roll} "
                  f"({n_roll / max(el, 1e-9):.1f}/s, {el:.0f}s)", flush=True)
    (a.out_dir / f"values.shard{a.shard_idx:02d}.manifest.json").write_text(json.dumps({
        "experiment": "step_value_labels_v1", "shard_idx": a.shard_idx, "num_shards": a.num_shards,
        "n_traces": n_all, "n_traces_this_run": len(traces), "n_rollouts_this_run": n_roll,
        "args": {k: str(v) for k, v in vars(a).items()},
        "trace_set_sha256": hashlib.sha256(a.trace_set.read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "seconds": time.perf_counter() - t0, "created": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
