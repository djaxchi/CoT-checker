#!/usr/bin/env python3
"""das_branch_subspace_v0 Phase 1b: boundary-patch ORACLE gate (GPU).

The decisive full-state test before any subspace search. For each fork we free-
generate K rollouts under three kinds of context and grade the final answer:

  correct    : question + prefix + golden step        (donor / correct branch)
  wrong      : question + prefix + rating -1 step      (base / wrong branch)
  oracle_L*  : the WRONG prompt, but with the residual state at the boundary token
               replaced (during prefill) by the CORRECT branch's boundary state at
               layer L. Full-state single-layer interchange.

Question (S6 Stage-0 asymmetry): S6 showed a full-state boundary patch recovers the
next-token distribution but ~0 of the immediately-elicited answer belief. Does that
next-token steering nonetheless compound over FREE generation into recovery of the
correct final answer? Recovery per layer = (oracle - wrong) / (correct - wrong).

A positive, control-beating oracle is the green light for the Phase-2 DAS subspace
search; a null oracle answers the hypothesis negatively at far less compute.

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python scripts/das_branch/das_oracle.py \
    --run_dir runs/causal_graph --out_dir runs/das_branch \
    --layers 12,20,26 --pilot_traces 200 --shard_id 0 --num_shards 4 --local_files_only
Merge (CPU):
  python scripts/das_branch/das_oracle.py --out_dir runs/das_branch --merge
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
from src.analysis.causal_graph import wilson_ci  # noqa: E402
from src.analysis.das_branch import (  # noqa: E402
    extract_boundary_state,
    fork_branch_prompts,
    generate_with_patch,
)
from src.analysis.transition_operator import stable_seed  # noqa: E402
from src.eval.math_grade import grade  # noqa: E402


def run(args, model, tok, device, rows: list) -> None:
    import torch

    traces = read_jsonl(Path(args.run_dir) / "traces_forks.jsonl")
    if args.pilot_traces:
        traces = traces[:args.pilot_traces]
    mine = [t for i, t in enumerate(traces) if i % args.num_shards == args.shard_id]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    t0 = time.perf_counter()
    for ji, tr in enumerate(mine):
        tid = tr["trace_id"]
        gt = tr.get("gt_answer")
        if not gt:
            continue
        prompts = fork_branch_prompts(tr)

        # donor states: one forward over the correct branch gives all layers at once
        cor_ids = tok(prompts["correct"], return_tensors="pt").to(device)["input_ids"]
        donor = {L: extract_boundary_state(model, cor_ids, L, cor_ids.shape[1] - 1)
                 for L in layers}

        conds = [("correct", prompts["correct"], None, None),
                 ("wrong", prompts["wrong"], None, None)]
        for L in layers:
            conds.append((f"oracle_L{L}", prompts["wrong"], L, donor[L]))

        for name, prompt, layer, state in conds:
            torch.manual_seed(stable_seed(tid + name, args.seed))
            conts = generate_with_patch(
                model, tok, device, prompt, args.k_rollouts, args.temperature,
                args.top_p, args.max_new_tokens, layer=layer, patched_state=state)
            for ri, cont in enumerate(conts):
                g = grade(cont, gt)
                rows.append({
                    "trace_id": tid, "split": tr.get("split", "test"),
                    "context": name, "rollout_idx": ri,
                    "correct": bool(g["correct"]), "gradeable": bool(g["gradeable"]),
                    "pred": (g["pred"] or "")[:60],
                    "has_alt_pos": bool(tr.get("alt_pos_step"))})
        if (ji + 1) % 5 == 0:
            el = time.perf_counter() - t0
            print(f"[oracle shard {args.shard_id}] {ji + 1}/{len(mine)} "
                  f"({el / (ji + 1):.1f}s/trace)", flush=True)


def merge(args) -> None:
    import pandas as pd
    from scipy import stats

    d = Path(args.out_dir)
    shards = sorted(d.glob("oracle_rollouts_shard*.parquet"))
    roll = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    roll.to_parquet(d / "oracle_rollouts.parquet")

    cur = (roll.groupby(["trace_id", "split", "context"])
           .agg(n=("correct", "size"), k=("correct", "sum"),
                gradeable_rate=("gradeable", "mean")).reset_index())
    cur["solve_rate"] = cur.k / cur.n
    cur.to_parquet(d / "oracle_curves.parquet")

    piv = cur.pivot_table(index="trace_id", columns="context",
                          values="solve_rate", aggfunc="first")
    layers = sorted(int(c.split("_L")[1]) for c in piv.columns if c.startswith("oracle_L"))
    out: dict = {"gradeable_rate_overall": float(roll.gradeable.mean()),
                 "n_traces": int(len(piv))}

    if {"correct", "wrong"}.issubset(piv.columns):
        both = piv[["correct", "wrong"]].dropna()
        dpre = both.correct - both.wrong
        out["premise_wrong_vs_correct"] = {
            "n": int(len(both)),
            "mean_correct": float(both.correct.mean()),
            "mean_wrong": float(both.wrong.mean()),
            "mean_gap_correct_minus_wrong": float(dpre.mean()),
            "p_wilcoxon_wrong_less": float(stats.wilcoxon(
                both.wrong, both.correct, alternative="less").pvalue)
            if len(both) >= 20 and (dpre != 0).any() else float("nan")}

    out["oracle_by_layer"] = {}
    for L in layers:
        col = f"oracle_L{L}"
        sub = piv[["correct", "wrong", col]].dropna()
        if len(sub) < 10:
            continue
        gap = (sub.correct - sub.wrong)
        rec = (sub[col] - sub.wrong)
        denom = gap.mean()
        out["oracle_by_layer"][f"L{L}"] = {
            "n": int(len(sub)),
            "mean_oracle": float(sub[col].mean()),
            "mean_oracle_minus_wrong": float(rec.mean()),
            "recovery_fraction": float(rec.mean() / denom) if denom > 0 else float("nan"),
            "p_wilcoxon_oracle_greater_wrong": float(stats.wilcoxon(
                sub[col], sub.wrong, alternative="greater").pvalue)
            if (rec != 0).any() else float("nan")}
    (d / "gates_oracle.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/causal_graph"))
    ap.add_argument("--out_dir", type=Path, default=Path("runs/das_branch"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layers", type=str, default="12,20,26")
    ap.add_argument("--k_rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--pilot_traces", type=int, default=200)
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
        args.out_dir / f"oracle_rollouts_shard{args.shard_id}.parquet")
    (args.out_dir / f"manifest_shard{args.shard_id}.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_rows": len(rows)}, indent=2))
    print(f"[oracle shard {args.shard_id}] rows {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
