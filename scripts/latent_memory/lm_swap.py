"""latent_memory_v0 test B: causal memory-swap. If the per-trace answer-optimised latent
is just a portable "the answer is X" code, injecting donor trace A's latent into recipient
trace B's context should pull B's answer onto A's gold; a norm-matched random vector should
not. If instead the latent carries reasoning state tied to B's own question, the donor pull
should be weak.

Per recipient B and donor A (different gold, same answer type), score a joint candidate set
[B_gold, A_gold, distractors] under four injections at the latent slot of [qB] SEP [P] SEP:
  self  = zB (sanity: recovers B_gold)   swap = zA (donor)   rand = random_like(zA) (control)
and a no-latent floor ([qB] SEP). Reports donor-win rate (A_gold outranks B_gold) and the
belief mass moved onto the donor answer, swap vs random vs floor.

    python -m scripts.latent_memory.lm_swap --traces runs/causal_graph/traces_forks.jsonl \
        --model_name_or_path Qwen/Qwen2.5-7B --local_files_only --layers 0,20 \
        --limit 200 --split test --run_dir runs/latent_memory_v0

Design: docs/latent_memory_v0_plan.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.causal_graph import ELICITATION_SUFFIX, cand_token_ids  # noqa: E402
from src.analysis.latent_memory import (  # noqa: E402
    belief_masses,
    chunk_pool_states,
    donor_win,
    joint_candidate_texts,
    latent_context_ids,
    optimize_latent,
    random_like,
)
from src.analysis.transition_operator import answer_type, normalize_answer  # noqa: E402
from scripts.latent_memory.lm_oracle import (  # noqa: E402
    build_trace_ids,
    read_jsonl,
    trace_candidates,
    valid_trace,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="runs/causal_graph/traces_forks.jsonl")
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--run_dir", type=Path, default=Path("runs/latent_memory_v0"))
    ap.add_argument("--layers", default="0,20")
    ap.add_argument("--m", type=int, default=1, help="latent width (swap uses m=1)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--pair_shift", type=int, default=1)
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
    suffix_ids = tok(ELICITATION_SUFFIX, add_special_tokens=False)["input_ids"]

    traces = [t for t in read_jsonl(Path(args.traces)) if valid_trace(t)]
    if args.split:
        traces = [t for t in traces if t.get("split") == args.split]
    if args.limit:
        traces = traces[: args.limit]

    # ---- per-trace prep: ids, own candidates, and optimised latent zB per layer -------
    prep = []
    for tr in traces:
        q_ids, step_ids, full_cot, no_cot, cot_lo = build_trace_ids(tok, tr, args.max_steps)
        gold_frac = "\\frac" in tr["gt_answer"]
        prep.append({
            "gold": tr["gt_answer"], "gold_n": normalize_answer(tr["gt_answer"], gold_frac),
            "gtype": answer_type(tr["gt_answer"]),
            "distractors": [c for c in trace_candidates(tr, args.k)][1:],
            "q_ids": q_ids, "full_cot": full_cot, "no_cot": no_cot,
            "cot_lo": cot_lo, "cot_hi": len(full_cot) - 1, "z": {},
        })

    def score(ctx_ids, cand_ids, layer=None, lo=0, hi=0, states=None):
        with torch.no_grad():
            if states is None:
                return candidate_mean_logprobs(model, ctx_ids, cand_ids, pad_id, device)
            return [float(s) for s in candidate_scores_grad(
                model, ctx_ids, cand_ids, pad_id, device, layer, lo, hi, states)]

    for layer in layers:
        for pr in prep:
            ids_t = torch.tensor([pr["full_cot"]], device=device)
            step_states = extract_span_states(model, ids_t, layer, pr["cot_lo"], pr["cot_hi"])
            lat_ctx, lo, hi = latent_context_ids(pr["q_ids"], args.m, pad_id)
            init = chunk_pool_states(step_states, args.m, "mean").to(device)
            # teacher belief over the trace's OWN candidates (gold-first) to optimise against
            own_cands = cand_token_ids(tok, ELICITATION_SUFFIX, [pr["gold_n"]] + pr["distractors"])
            t_scores = score(pr["full_cot"] + suffix_ids, own_cands)
            teacher_belief = torch.softmax(torch.tensor(t_scores), dim=-1)
            res = optimize_latent(model, lat_ctx, lo, hi, layer, own_cands, suffix_ids,
                                  teacher_belief, init, pad_id, device,
                                  epochs=args.epochs, lr=args.lr)
            pr["z"][layer] = res.z.to(device)

    # ---- pairing + swap scoring ------------------------------------------------------
    n = len(prep)
    rows = []
    for layer in layers:
        for i in range(n):
            B = prep[i]
            # first eligible donor: type-matched, different gold (scan offsets so nearly
            # every recipient is paired, not just a fixed shift)
            A = None
            for off in range(args.pair_shift, n):
                cand = prep[(i + off) % n]
                if cand["gtype"] == B["gtype"] and cand["gold_n"] != B["gold_n"]:
                    A = cand
                    break
            if A is None:
                continue
            texts, ib, ia = joint_candidate_texts(B["gold"], A["gold"],
                                                  B["distractors"], A["distractors"], args.k)
            if ib == ia:
                continue
            cand_ids = cand_token_ids(tok, ELICITATION_SUFFIX, texts)
            lat_ctx, lo, hi = latent_context_ids(B["q_ids"], args.m, pad_id)
            ctx = lat_ctx + suffix_ids
            zB, zA = B["z"][layer], A["z"][layer]
            s_self = score(ctx, cand_ids, layer, lo, hi, zB)
            s_swap = score(ctx, cand_ids, layer, lo, hi, zA)
            s_rand = score(ctx, cand_ids, layer, lo, hi, random_like(zA, args.seed + i))
            s_floor = score(B["no_cot"] + suffix_ids, cand_ids)
            mb_self, ma_self = belief_masses(s_self, ib, ia)
            mb_swap, ma_swap = belief_masses(s_swap, ib, ia)
            mb_rand, ma_rand = belief_masses(s_rand, ib, ia)
            mb_floor, ma_floor = belief_masses(s_floor, ib, ia)
            rows.append({
                "layer": layer, "recipient": B["gold_n"], "donor": A["gold_n"],
                "mb_self": mb_self, "ma_self": ma_self,
                "mb_swap": mb_swap, "ma_swap": ma_swap,
                "mb_rand": mb_rand, "ma_rand": ma_rand,
                "mb_floor": mb_floor, "ma_floor": ma_floor,
                "donor_win_swap": donor_win(ma_swap, mb_swap),
                "donor_win_rand": donor_win(ma_rand, mb_rand),
                "self_win": donor_win(mb_self, ma_self),  # sanity: B keeps its own answer
            })

    args.run_dir.mkdir(parents=True, exist_ok=True)
    out = args.run_dir / "swap_rows.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows))

    print(f"pairs={len(rows)}")
    for layer in layers:
        L = [r for r in rows if r["layer"] == layer]
        if not L:
            continue
        agg = {
            "n": len(L),
            "self_win_rate": mean(r["self_win"] for r in L),
            "donor_win_rate_swap": mean(r["donor_win_swap"] for r in L),
            "donor_win_rate_rand": mean(r["donor_win_rand"] for r in L),
            "ma_floor": mean(r["ma_floor"] for r in L),
            "ma_swap": mean(r["ma_swap"] for r in L),
            "ma_rand": mean(r["ma_rand"] for r in L),
            "mb_floor": mean(r["mb_floor"] for r in L),
            "mb_swap": mean(r["mb_swap"] for r in L),
            "mb_self": mean(r["mb_self"] for r in L),
        }
        print(f"\n-- layer {layer} (n={agg['n']}) --")
        print(f"  self-win rate (B keeps own answer w/ zB): {agg['self_win_rate']:.3f}")
        print(f"  donor-win rate  swap zA : {agg['donor_win_rate_swap']:.3f}"
              f"   random control : {agg['donor_win_rate_rand']:.3f}")
        print(f"  donor-answer mass  floor {agg['ma_floor']:.3f} -> swap "
              f"{agg['ma_swap']:.3f}  (rand {agg['ma_rand']:.3f})")
        print(f"  recip-answer mass  self {agg['mb_self']:.3f}  swap {agg['mb_swap']:.3f}"
              f"  floor {agg['mb_floor']:.3f}")
        (args.run_dir / f"swap_summary_L{layer}.json").write_text(json.dumps(agg, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
