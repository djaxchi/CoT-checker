"""latent_memory_v0 test A: trace-probe falsification of the answer-shortcut. The oracle
showed a single latent, optimised on the final-ANSWER belief, recovers ~1.0 of it. Does
that same latent also let the frozen model answer a DIFFERENT question about the trace,
namely recall an intermediate value computed mid-reasoning?

For each trace: pick a clean intermediate integer (appears in a middle step, not in the
question, != the final answer). Then measure recovery of that intermediate under:
  - the answer-optimised latent zB       (if ~0 -> zB is an answer code, shortcut)
  - a latent optimised ON the probe z_p  (capacity control; if ~1.0 the value IS injectable)
against full-CoT (ceiling, the value is in context) and no-CoT (floor). Each row also
carries zB's ANSWER recovery, so the contrast answer_rec(zB) >> probe_rec(zB) ~ shortcut,
while probe_rec(z_p) ~ answer_rec(zB) shows the info was injectable, isolating the shortcut.

    python -m scripts.latent_memory.lm_probe --traces runs/causal_graph/traces_forks.jsonl \
        --model_name_or_path Qwen/Qwen2.5-7B --local_files_only --layers 0,20 \
        --limit 200 --split test --run_dir runs/latent_memory_v0

Design: docs/latent_memory_v0_plan.md.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import median

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.causal_graph import ELICITATION_SUFFIX, cand_token_ids  # noqa: E402
from src.analysis.latent_memory import (  # noqa: E402
    PROBE_SUFFIX,
    chunk_pool_states,
    latent_context_ids,
    optimize_latent,
    pick_probe_target,
    recovery,
)
from src.analysis.transition_operator import integer_perturbations  # noqa: E402
from scripts.latent_memory.lm_oracle import (  # noqa: E402
    build_trace_ids,
    read_jsonl,
    trace_candidates,
    valid_trace,
)


def probe_candidates(gold: str, rng: random.Random, k: int) -> list[str]:
    """Gold-first probe candidate integers: gold + near perturbations (hard distractors)."""
    out, seen = [gold], {gold}
    for c in integer_perturbations(gold, rng, k * 3):
        if c not in seen:
            seen.add(c); out.append(c)
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="runs/causal_graph/traces_forks.jsonl")
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--run_dir", type=Path, default=Path("runs/latent_memory_v0"))
    ap.add_argument("--layers", default="0,20")
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.analysis.das_span import extract_span_states
    from src.analysis.latent_memory import candidate_scores_grad
    from src.analysis.transition_operator import candidate_mean_logprobs

    layers = [int(x) for x in args.layers.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    pad_id = tok.pad_token_id or tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ans_suffix = tok(ELICITATION_SUFFIX, add_special_tokens=False)["input_ids"]
    probe_suffix = tok(PROBE_SUFFIX, add_special_tokens=False)["input_ids"]
    rng = random.Random(args.seed)

    traces = [t for t in read_jsonl(Path(args.traces)) if valid_trace(t)]
    if args.split:
        traces = [t for t in traces if t.get("split") == args.split]
    if args.limit:
        traces = traces[: args.limit]

    def margin(ctx_ids, cand_ids, layer=None, lo=0, hi=0, states=None):
        with torch.no_grad():
            if states is None:
                s = candidate_mean_logprobs(model, ctx_ids, cand_ids, pad_id, device)
            else:
                s = [float(x) for x in candidate_scores_grad(
                    model, ctx_ids, cand_ids, pad_id, device, layer, lo, hi, states)]
        return s[0] - max(s[1:])

    def belief(ctx_ids, cand_ids):
        with torch.no_grad():
            s = candidate_mean_logprobs(model, ctx_ids, cand_ids, pad_id, device)
        return torch.softmax(torch.tensor(s), dim=-1)

    rows = []
    for ti, tr in enumerate(traces):
        tgt = pick_probe_target(tr["question"], tr["steps"], tr["gt_answer"])
        if tgt is None:
            continue
        gold_int, step_idx = tgt
        q_ids, step_ids, full_cot, no_cot, cot_lo = build_trace_ids(tok, tr, args.max_steps)
        cot_hi = len(full_cot) - 1

        ans_txt = trace_candidates(tr, args.k)          # gold-first answer candidates
        ans_cand = cand_token_ids(tok, ELICITATION_SUFFIX, ans_txt)
        prb_txt = probe_candidates(gold_int, rng, args.k)
        prb_cand = cand_token_ids(tok, PROBE_SUFFIX, prb_txt)

        # ceilings/floors (no latent) for both questions
        ans_full = margin(full_cot + ans_suffix, ans_cand)
        ans_no = margin(no_cot + ans_suffix, ans_cand)
        prb_full = margin(full_cot + probe_suffix, prb_cand)
        prb_no = margin(no_cot + probe_suffix, prb_cand)

        ans_belief = belief(full_cot + ans_suffix, ans_cand)
        prb_belief = belief(full_cot + probe_suffix, prb_cand)

        for layer in layers:
            ids_t = torch.tensor([full_cot], device=device)
            step_states = extract_span_states(model, ids_t, layer, cot_lo, cot_hi)
            lat_ctx, lo, hi = latent_context_ids(q_ids, args.m, pad_id)
            init = chunk_pool_states(step_states, args.m, "mean").to(device)

            zB = optimize_latent(model, lat_ctx, lo, hi, layer, ans_cand, ans_suffix,
                                 ans_belief, init, pad_id, device,
                                 epochs=args.epochs, lr=args.lr).z.to(device)
            zP = optimize_latent(model, lat_ctx, lo, hi, layer, prb_cand, probe_suffix,
                                 prb_belief, init, pad_id, device,
                                 epochs=args.epochs, lr=args.lr).z.to(device)

            ans_zB = margin(lat_ctx + ans_suffix, ans_cand, layer, lo, hi, zB)
            prb_zB = margin(lat_ctx + probe_suffix, prb_cand, layer, lo, hi, zB)
            prb_zP = margin(lat_ctx + probe_suffix, prb_cand, layer, lo, hi, zP)

            rows.append({
                "trace_id": tr.get("trace_id", ti), "layer": layer,
                "gold_int": gold_int, "step_idx": step_idx, "n_steps": len(step_ids),
                "answer_rec_zB": recovery(ans_zB, ans_no, ans_full),
                "probe_rec_zB": recovery(prb_zB, prb_no, prb_full),
                "probe_rec_zP": recovery(prb_zP, prb_no, prb_full),
                "ans_full": ans_full, "ans_no": ans_no,
                "prb_full": prb_full, "prb_no": prb_no,
            })
        if (ti + 1) % 20 == 0:
            print(f"[probe] {ti + 1}/{len(traces)} traces, {len(rows)} rows")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    out = args.run_dir / "probe_rows.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows))

    def med(xs):
        xs = [x for x in xs if x == x]
        return median(xs) if xs else float("nan")

    print(f"\nrows={len(rows)}")
    for layer in layers:
        L = [r for r in rows if r["layer"] == layer]
        if not L:
            continue
        agg = {
            "n": len(L),
            "answer_rec_zB": med([r["answer_rec_zB"] for r in L]),
            "probe_rec_zB": med([r["probe_rec_zB"] for r in L]),
            "probe_rec_zP": med([r["probe_rec_zP"] for r in L]),
            "prb_full_margin": med([r["prb_full"] for r in L]),
            "prb_no_margin": med([r["prb_no"] for r in L]),
        }
        print(f"\n-- layer {layer} (n={agg['n']}) --")
        print(f"  answer recovery, answer-latent zB : {agg['answer_rec_zB']:.3f}  (shortcut carrier)")
        print(f"  PROBE  recovery, answer-latent zB : {agg['probe_rec_zB']:.3f}  (KEY: ~0 => shortcut)")
        print(f"  PROBE  recovery, probe-latent  zP : {agg['probe_rec_zP']:.3f}  (capacity control)")
        print(f"  probe margins: full-CoT {agg['prb_full_margin']:.3f}  no-CoT {agg['prb_no_margin']:.3f}")
        (args.run_dir / f"probe_summary_L{layer}.json").write_text(json.dumps(agg, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
