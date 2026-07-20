#!/usr/bin/env python3
"""das_branch_subspace_v0 whole-step-span ORACLE gate (GPU).

Successor to the null boundary oracle (Phase 1b). Injects the CORRECT branch's full
candidate-step residual span into the WRONG branch at a layer, so the upper blocks
regenerate K/V from the patched span, and asks whether the answer moves toward the
correct branch. Primary readout is the teacher-forced gold-answer margin (the
deterministic, continuous extension of S6-Stage0's answer-belief measure, which was
~0 at the boundary); --mode freegen adds free-generation solve probability.

Alignment (fork span lengths usually differ):
  --align equal : forks whose correct/wrong steps have identical token length
                  (RoPE-clean, whole span, ~25 forks).
  --align lastk : the final k positions of each step, all forks (powered, but donor
                  and injection positions differ when lengths differ).

Conditions: wrong (unpatched base), correct (unpatched source), oracle_L{layers}
(correct span into wrong), xspan_L{ctrl_layer} (a different fork's donor span, the
generic-perturbation control). Recovery_L = (oracle - wrong) / (correct - wrong).

Usage (one shard, tf margins):
  CUDA_VISIBLE_DEVICES=0 python scripts/das_branch/das_span_oracle.py \
    --run_dir runs/causal_graph --out_dir runs/das_span --align lastk --k 8 \
    --layers 12,20,26 --shard_id 0 --num_shards 4 --local_files_only
Merge (CPU):
  python scripts/das_branch/das_span_oracle.py --out_dir runs/das_span \
    --align lastk --merge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl  # noqa: E402
from src.analysis.das_span import (  # noqa: E402
    aligned_positions,
    extract_span_states,
    fork_span_ids,
    generate_with_span_patch,
    gold_margin,
    span_candidate_logprobs,
    suffix_ids,
)
from src.analysis.transition_operator import stable_seed  # noqa: E402
from src.eval.math_grade import grade  # noqa: E402

MIN_SPAN = 4  # skip degenerate short steps (last-k needs something to inject)


def select(traces, tok, align, k, min_span=MIN_SPAN):
    """Yield (tr, wrong_ids, wrong_span, correct_ids, correct_span, inj, don)."""
    out = []
    for tr in traces:
        if not tr.get("gt_answer"):
            continue
        try:
            wids, wlo, whi = fork_span_ids(tok, tr, "wrong")
            cids, clo, chi = fork_span_ids(tok, tr, "correct")
        except Exception:
            continue
        wn, cn = whi - wlo, chi - clo
        if wn < min_span or cn < min_span:
            continue
        if align == "equal" and wn != cn:
            continue
        ilo, ihi, dlo, dhi = aligned_positions((wlo, whi), (clo, chi), align, k)
        out.append((tr, wids, (wlo, whi), cids, (clo, chi), (ilo, ihi), (dlo, dhi)))
    return out


def run(args, model, tok, device, rows: list) -> None:
    import torch

    traces = read_jsonl(Path(args.run_dir) / "traces_forks.jsonl")
    if args.max_traces:
        traces = traces[:args.max_traces]
    jobs = select(traces, tok, args.align, args.k)
    jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard_id]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    ctrl_layer = layers[len(layers) // 2]
    sfx = suffix_ids(tok)
    pad = tok.pad_token_id or tok.eos_token_id
    prev_donor = {}  # layer -> last fork's donor states, for the xspan control

    t0 = time.perf_counter()
    for ji, (tr, wids, wspan, cids, cspan, inj, don) in enumerate(jobs):
        tid = tr["trace_id"]
        cand_ids = [tok(c, add_special_tokens=False)["input_ids"]
                    for c in tr["candidates"]]
        cor_t = torch.tensor([cids], device=device)
        donor = {L: extract_span_states(model, cor_t, L, don[0], don[1])
                 for L in layers}
        ilo, ihi = inj

        def margin(ctx_ids, layer=None, states=None):
            sc = span_candidate_logprobs(model, ctx_ids + sfx, cand_ids, pad, device,
                                         layer=layer, lo=ilo, hi=ihi, states=states)
            return gold_margin(sc)

        rec = {"trace_id": tid, "split": tr.get("split", "test"), "align": args.align,
               "wrong_span": wspan[1] - wspan[0], "correct_span": cspan[1] - cspan[0],
               "inj_width": ihi - ilo,
               "m_wrong": margin(wids), "m_correct": margin(cids)}
        for L in layers:
            rec[f"m_oracle_L{L}"] = margin(wids, layer=L, states=donor[L])
        if ctrl_layer in prev_donor and prev_donor[ctrl_layer].shape[0] == (ihi - ilo):
            rec[f"m_xspan_L{ctrl_layer}"] = margin(
                wids, layer=ctrl_layer, states=prev_donor[ctrl_layer])
        prev_donor = {L: donor[L] for L in layers}

        if args.mode == "freegen":
            torch.manual_seed(stable_seed(tid + "wrong", args.seed))
            gw = [grade(c, tr["gt_answer"])["correct"] for c in generate_with_span_patch(
                model, tok, device, wids, args.k_rollouts, args.temperature,
                args.top_p, args.max_new_tokens)]
            rec["solve_wrong"] = sum(gw) / len(gw)
            torch.manual_seed(stable_seed(tid + "correct", args.seed))
            gc = [grade(c, tr["gt_answer"])["correct"] for c in generate_with_span_patch(
                model, tok, device, cids, args.k_rollouts, args.temperature,
                args.top_p, args.max_new_tokens)]
            rec["solve_correct"] = sum(gc) / len(gc)
            for L in layers:
                torch.manual_seed(stable_seed(tid + f"o{L}", args.seed))
                go = [grade(c, tr["gt_answer"])["correct"]
                      for c in generate_with_span_patch(
                          model, tok, device, wids, args.k_rollouts, args.temperature,
                          args.top_p, args.max_new_tokens, layer=L, lo=ilo, hi=ihi,
                          states=donor[L])]
                rec[f"solve_oracle_L{L}"] = sum(go) / len(go)
        rows.append(rec)
        if (ji + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"[span {args.align} shard {args.shard_id}] {ji + 1}/{len(jobs)} "
                  f"({el / (ji + 1):.1f}s/fork)", flush=True)


def merge(args) -> None:
    import numpy as np
    import pandas as pd
    from scipy import stats

    d = Path(args.out_dir)
    shards = sorted(d.glob(f"span_{args.align}_shard*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    df.to_parquet(d / f"span_{args.align}.parquet")
    layers = sorted(int(c.split("_L")[1]) for c in df.columns
                    if c.startswith("m_oracle_L"))

    def paired(a, b, alt):
        m = df[[a, b]].dropna()
        if len(m) < 10 or not (m[a] - m[b]).abs().gt(0).any():
            return {"n": int(len(m)), "p": float("nan")}
        return {"n": int(len(m)), "p": float(stats.wilcoxon(
            m[a], m[b], alternative=alt).pvalue)}

    gap = (df.m_correct - df.m_wrong)
    out = {"align": args.align, "n": int(len(df)),
           "belief_gap_correct_minus_wrong": {
               "mean": float(gap.mean()), "median": float(gap.median()),
               **paired("m_correct", "m_wrong", "greater")},
           "margin_by_layer": {}}
    for L in layers:
        col = f"m_oracle_L{L}"
        sub = df[["m_correct", "m_wrong", col]].dropna()
        delta = (sub[col] - sub.m_wrong)
        denom = (sub.m_correct - sub.m_wrong).mean()
        out["margin_by_layer"][f"L{L}"] = {
            "mean_oracle_minus_wrong": float(delta.mean()),
            "recovery_fraction": float(delta.mean() / denom) if denom != 0 else float("nan"),
            **paired(col, "m_wrong", "greater")}
        xs = f"m_xspan_L{L}"
        if xs in df.columns:
            out["margin_by_layer"][f"L{L}"]["xspan_mean_minus_wrong"] = float(
                (df[xs] - df.m_wrong).dropna().mean())

    if any(c.startswith("solve_oracle_L") for c in df.columns):
        sg = (df.solve_correct - df.solve_wrong)
        out["freegen"] = {"mean_solve_wrong": float(df.solve_wrong.mean()),
                          "mean_solve_correct": float(df.solve_correct.mean()),
                          "mean_gap": float(sg.mean())}
        for L in layers:
            col = f"solve_oracle_L{L}"
            if col in df.columns:
                delta = (df[col] - df.solve_wrong)
                denom = sg.mean()
                out["freegen"][f"L{L}"] = {
                    "mean_oracle_minus_wrong": float(delta.mean()),
                    "recovery_fraction": float(delta.mean() / denom)
                    if denom != 0 else float("nan"),
                    **paired(col, "solve_wrong", "greater")}
    (d / f"gates_span_{args.align}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/causal_graph"))
    ap.add_argument("--out_dir", type=Path, default=Path("runs/das_span"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--align", choices=["equal", "lastk"], default="lastk")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--layers", type=str, default="12,20,26")
    ap.add_argument("--mode", choices=["tf_margin", "freegen"], default="tf_margin")
    ap.add_argument("--k_rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_traces", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args)
        return

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only).to(device).eval()

    rows: list = []
    run(args, model, tok, device, rows)
    pd.DataFrame(rows).to_parquet(
        args.out_dir / f"span_{args.align}_shard{args.shard_id}.parquet")
    (args.out_dir / f"manifest_{args.align}_shard{args.shard_id}.json").write_text(
        json.dumps({"created": datetime.now(timezone.utc).isoformat(),
                    "git_commit": git_commit(),
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "n_rows": len(rows)}, indent=2))
    print(f"[span {args.align} shard {args.shard_id}] rows {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
