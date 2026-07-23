"""latent_memory_v0 multi-target reasoning-state oracle. Tests A/B proved a latent optimised
on the final ANSWER is an answer-shortcut (recalls no intermediate value). The real question:
can ONE latent, optimised JOINTLY on a distribution of queries (the final answer plus several
intermediate values, each anchored to a distinct textual cue), serve them all, and how does
that scale with latent width m?

Per trace: build the answer query + up to n_targets cue-anchored intermediate queries. For
each width m and layer, jointly optimise one latent on all queries (sum of KL to the full-CoT
teacher belief) and record per-query recovery of the full-CoT margin. If mean intermediate
recovery is ~1.0 at m=1, the reasoning state compresses into a single vector (the answer-only
objective just missed it). If it climbs with m, capacity scales with the number of facts.

    python -m scripts.latent_memory.lm_multitarget --traces runs/causal_graph/traces_forks.jsonl \
        --model_name_or_path Qwen/Qwen2.5-7B --local_files_only --layers 0,20 \
        --m_list 1,2,4,8,16 --n_targets 3 --limit 200 --split test \
        --run_dir runs/latent_memory_v0 --shard_id 0 --num_shards 4

Design: docs/latent_memory_v0_plan.md.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.causal_graph import ELICITATION_SUFFIX, cand_token_ids  # noqa: E402
from src.analysis.latent_memory import (  # noqa: E402
    chunk_pool_states,
    latent_context_ids,
    optimize_latent_multi,
    pick_probe_targets,
    recovery,
)
from scripts.latent_memory.lm_oracle import (  # noqa: E402
    build_trace_ids,
    read_jsonl,
    trace_candidates,
    valid_trace,
)
from scripts.latent_memory.lm_probe import probe_candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="runs/causal_graph/traces_forks.jsonl")
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--run_dir", type=Path, default=Path("runs/latent_memory_v0"))
    ap.add_argument("--layers", default="0,20")
    ap.add_argument("--m_list", default="1,2,4,8,16")
    ap.add_argument("--n_targets", type=int, default=3, help="intermediate queries per trace")
    ap.add_argument("--min_targets", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.analysis.das_span import extract_span_states
    from src.analysis.transition_operator import candidate_mean_logprobs

    layers = [int(x) for x in args.layers.split(",")]
    m_list = [int(x) for x in args.m_list.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    pad_id = tok.pad_token_id or tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rng = random.Random(args.seed)

    traces = [t for t in read_jsonl(Path(args.traces)) if valid_trace(t)]
    if args.split:
        traces = [t for t in traces if t.get("split") == args.split]
    if args.limit:
        traces = traces[: args.limit]
    mine = traces[args.shard_id :: args.num_shards]

    def margin(ctx_ids, cand_ids):
        with torch.no_grad():
            s = candidate_mean_logprobs(model, ctx_ids, cand_ids, pad_id, device)
        return s, s[0] - max(s[1:])

    rows = []
    for ti, tr in enumerate(mine):
        inters = pick_probe_targets(tr["question"], tr["steps"], tr["gt_answer"],
                                    n_targets=args.n_targets)
        if len(inters) < args.min_targets:
            continue
        q_ids, step_ids, full_cot, no_cot, cot_lo = build_trace_ids(tok, tr, args.max_steps)
        cot_hi = len(full_cot) - 1

        # queries: [answer] + intermediates, each with suffix ids + gold-first candidates
        queries = []
        ans_suffix = tok(ELICITATION_SUFFIX, add_special_tokens=False)["input_ids"]
        queries.append({"name": "answer", "suffix_ids": ans_suffix,
                        "cand_ids": cand_token_ids(tok, ELICITATION_SUFFIX,
                                                   trace_candidates(tr, args.k))})
        for it in inters:
            suf = "\n" + it["cue"]
            queries.append({"name": f"inter@{it['step_idx']}",
                            "suffix_ids": tok(suf, add_special_tokens=False)["input_ids"],
                            "cand_ids": cand_token_ids(tok, suf,
                                                       probe_candidates(it["gold"], rng, args.k))})

        # per-query ceiling (full-CoT), floor (no-CoT), and teacher belief
        ok = True
        for q in queries:
            fs, fm = margin(full_cot + q["suffix_ids"], q["cand_ids"])
            _, nm = margin(no_cot + q["suffix_ids"], q["cand_ids"])
            q["full"], q["no"] = fm, nm
            q["teacher_belief"] = torch.softmax(torch.tensor(fs), dim=-1)
            if abs(fm - nm) < 1e-6:   # degenerate query (no gap to recover)
                ok = False
        if not ok:
            continue

        for layer in layers:
            ids_t = torch.tensor([full_cot], device=device)
            step_states = extract_span_states(model, ids_t, layer, cot_lo, cot_hi)
            for m in m_list:
                lat_ctx, lo, hi = latent_context_ids(q_ids, m, pad_id)
                init = chunk_pool_states(step_states, m, "mean").to(device)
                res = optimize_latent_multi(model, lat_ctx, lo, hi, layer, queries, init,
                                            pad_id, device, epochs=args.epochs, lr=args.lr)
                recs = [recovery(res["margins"][j], queries[j]["no"], queries[j]["full"])
                        for j in range(len(queries))]
                ans_rec = recs[0]
                inter = [r for r in recs[1:] if r == r]
                rows.append({
                    "trace_id": tr.get("trace_id", ti), "layer": layer, "m": m,
                    "n_inter": len(inter), "answer_rec": ans_rec,
                    "inter_rec_mean": sum(inter) / len(inter) if inter else float("nan"),
                    "inter_rec_min": min(inter) if inter else float("nan"),
                    "inter_recs": inter,
                    "loss0": res["loss_history"][0], "lossT": res["loss_history"][-1],
                })
        if (ti + 1) % 10 == 0:
            print(f"[mt shard {args.shard_id}] {ti + 1}/{len(mine)} traces, {len(rows)} rows")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    out = args.run_dir / f"multitarget_shard{args.shard_id}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"[mt shard {args.shard_id}] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
