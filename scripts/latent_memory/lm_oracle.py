"""latent_memory_v0 capacity oracle: per-trace, how few free latent vectors injected at
layer L reproduce the full-CoT answer belief on the frozen base model?

For each trace: teacher (full golden CoT) and floor (question only) answer-belief margins,
fixed chunk-pool baselines, and an optimised latent-memory oracle, swept over width m and
injection layer. Writes one JSONL row per (trace, layer, m, method).

    python -m scripts.latent_memory.lm_oracle \
        --traces runs/causal_graph/traces_forks.jsonl \
        --model_name_or_path Qwen/Qwen2.5-7B --local_files_only \
        --layers 0,20 --m_list 1,2,4,8,16,32 --limit 300 --split test \
        --run_dir runs/latent_memory_v0 --shard_id 0 --num_shards 1

Design + rationale: docs/latent_memory_v0_plan.md. Reuses das_span injection, the S6
elicitation-suffix candidate machinery, and the causal_graph fork traces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.causal_graph import (  # noqa: E402
    ELICITATION_SUFFIX,
    cand_token_ids,
    encode_pieces,
)
from src.analysis.latent_memory import (  # noqa: E402
    chunk_pool_states,
    full_cot_context_ids,
    latent_context_ids,
    no_cot_context_ids,
    optimize_latent,
    recovery,
)
from src.analysis.transition_operator import SEP_TOKEN_ID, build_candidates


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def valid_trace(tr: dict) -> bool:
    return bool(tr.get("question")) and bool(tr.get("steps")) and bool(tr.get("gt_answer"))


def trace_candidates(tr: dict, k: int) -> list[str]:
    """Gold-first candidate answers: reuse the trace's pre-built set if present, else
    build one from gt_answer + wrong_finals + pre_generated_answer."""
    cands = tr.get("candidates")
    if cands and len(cands) >= 2:
        return cands[:k]
    return build_candidates(tr["gt_answer"], tr.get("pre_generated_answer"),
                            tr.get("wrong_finals", []), k=k)


def build_trace_ids(tok, tr: dict, max_steps: int | None):
    """(question_ids, step_ids, full_cot_ids, no_cot_ids, cot_lo)."""
    steps = tr["steps"] if max_steps is None else tr["steps"][:max_steps]
    q_ids = encode_pieces(tok, [tr["question"]])[0]
    step_ids = encode_pieces(tok, steps)
    full_cot = full_cot_context_ids(q_ids, step_ids)
    no_cot = no_cot_context_ids(q_ids)
    cot_lo = len(q_ids) + 1  # first step-region position (after question + SEP)
    return q_ids, step_ids, full_cot, no_cot, cot_lo


def dry_run(args, tok=None):
    """Validate data plumbing and token budgets without loading the model."""
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                            local_files_only=args.local_files_only)
    traces = [t for t in read_jsonl(Path(args.traces)) if valid_trace(t)]
    if args.split:
        traces = [t for t in traces if t.get("split") == args.split]
    traces = traces[: args.limit] if args.limit else traces
    lengths, n_cands, n_steps = [], [], []
    for tr in traces:
        _, step_ids, full_cot, _, _ = build_trace_ids(tok, tr, args.max_steps)
        lengths.append(len(full_cot))
        n_steps.append(len(step_ids))
        n_cands.append(len(trace_candidates(tr, args.k)))
    if not lengths:
        print("[dry_run] no valid traces after filtering"); return
    lengths.sort()
    print(f"[dry_run] traces={len(traces)} split={args.split}")
    print(f"[dry_run] full-CoT token len: min={lengths[0]} "
          f"median={lengths[len(lengths)//2]} p90={lengths[int(0.9*len(lengths))]} "
          f"max={lengths[-1]}")
    print(f"[dry_run] steps/trace: min={min(n_steps)} max={max(n_steps)}")
    print(f"[dry_run] candidates/trace: min={min(n_cands)} max={max(n_cands)}")
    print(f"[dry_run] layers={args.layers} m_list={args.m_list} suffix='{ELICITATION_SUFFIX}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="runs/causal_graph/traces_forks.jsonl")
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--run_dir", type=Path, default=Path("runs/latent_memory_v0"))
    ap.add_argument("--layers", default="0,20", help="comma list; 0=soft input tokens")
    ap.add_argument("--m_list", default="1,2,4,8,16,32")
    ap.add_argument("--k", type=int, default=8, help="candidate-set size")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap CoT length (keeps very long traces tractable)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    m_list = [int(x) for x in args.m_list.split(",")]

    if args.dry_run:
        dry_run(args); return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.analysis.das_span import extract_span_states
    from src.analysis.latent_memory import candidate_scores_grad

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    pad_id = tok.pad_token_id or tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    suffix_ids = tok(ELICITATION_SUFFIX, add_special_tokens=False)["input_ids"]

    traces = [t for t in read_jsonl(Path(args.traces)) if valid_trace(t)]
    if args.split:
        traces = [t for t in traces if t.get("split") == args.split]
    if args.limit:
        traces = traces[: args.limit]
    mine = traces[args.shard_id :: args.num_shards]

    args.run_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.run_dir / f"oracle_shard{args.shard_id}.jsonl"
    rows: list[dict] = []

    def score_no_grad(ctx_ids, cands, layer=1, lo=0, hi=0, states=None):
        with torch.no_grad():
            st = states if states is not None else torch.zeros(0, model.config.hidden_size,
                                                               device=device)
            return [float(s) for s in candidate_scores_grad(
                model, ctx_ids, cands, pad_id, device, layer, lo, hi, st)]

    for ti, tr in enumerate(mine):
        q_ids, step_ids, full_cot, no_cot, cot_lo = build_trace_ids(tok, tr, args.max_steps)
        cands_txt = trace_candidates(tr, args.k)
        cand_ids = cand_token_ids(tok, ELICITATION_SUFFIX, cands_txt)

        t_scores = score_no_grad(full_cot + suffix_ids, cand_ids)
        margin_full = t_scores[0] - max(t_scores[1:])
        teacher_belief = torch.softmax(torch.tensor(t_scores), dim=-1)
        n_scores = score_no_grad(no_cot + suffix_ids, cand_ids)
        margin_no = n_scores[0] - max(n_scores[1:])

        base = {"trace_id": tr.get("trace_id", ti), "split": tr.get("split"),
                "n_steps": len(step_ids), "cot_len": len(full_cot),
                "margin_full": margin_full, "margin_no": margin_no,
                "k": len(cand_ids)}
        rows.append({**base, "layer": None, "m": None, "method": "teacher_full",
                     "margin": margin_full, "recovery": 1.0})
        rows.append({**base, "layer": None, "m": None, "method": "no_cot",
                     "margin": margin_no, "recovery": 0.0})

        for layer in layers:
            # step residuals at this injection layer for pooling init / baselines
            ids_t = torch.tensor([full_cot], device=device)
            cot_hi = len(full_cot) - 1  # exclude the trailing SEP
            step_states = extract_span_states(model, ids_t, layer, cot_lo, cot_hi)
            for m in m_list:
                lat_ctx, lo, hi = latent_context_ids(q_ids, m, pad_id)
                for mode in ("mean", "max"):
                    pooled = chunk_pool_states(step_states, m, mode).to(device)
                    s = score_no_grad(lat_ctx + suffix_ids, cand_ids, layer, lo, hi, pooled)
                    mg = s[0] - max(s[1:])
                    rows.append({**base, "layer": layer, "m": m,
                                 "method": f"pool_{mode}", "margin": mg,
                                 "recovery": recovery(mg, margin_no, margin_full)})
                init = chunk_pool_states(step_states, m, "mean").to(device)
                res = optimize_latent(model, lat_ctx, lo, hi, layer, cand_ids,
                                      suffix_ids, teacher_belief, init, pad_id, device,
                                      epochs=args.epochs, lr=args.lr)
                rows.append({**base, "layer": layer, "m": m, "method": "oracle",
                             "margin": res.margin,
                             "recovery": recovery(res.margin, margin_no, margin_full),
                             "loss0": res.loss_history[0],
                             "lossT": res.loss_history[-1]})
        if (ti + 1) % 10 == 0:
            print(f"[shard {args.shard_id}] {ti + 1}/{len(mine)} traces")
            out_path.write_text("\n".join(json.dumps(r) for r in rows))

    out_path.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"[shard {args.shard_id}] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
