#!/usr/bin/env python3
"""das_branch_subspace_v0: behavioural test of the learned DAS subspace (GPU).

The margin-regression fit showed a k=8 subspace at L12 recovers ~25% of the answer-
belief gap on held-out forks and beats random. This asks the behavioural question:
does that same learned subspace move the FREE-GENERATION solve rate, not just the
teacher-forced margin? (The whole-span oracle transferred belief 88% but solving only
35%, so behaviour must be checked separately.)

For held-out (val) forks, free-generate K rollouts under each condition and grade:
  wrong   : wrong-branch context, unpatched            (base)
  correct : correct-branch context, unpatched          (donor / ceiling reference)
  oracle  : wrong ctx, full correct span injected at L  (full-state ceiling)
  das     : wrong ctx, U U^T interchange of the span    (learned k-dim subspace)
  random  : wrong ctx, same-k RANDOM subspace interchange (control)

Recovery = (solve_cond - solve_wrong) / (solve_correct - solve_wrong).

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python scripts/das_branch/das_subspace_freegen.py \
    --run_dir runs/causal_graph --out_dir runs/das_train --layer 12 --k_sub 8 \
    --seed 0 --shard_id 0 --num_shards 4 --local_files_only
Merge:
  python scripts/das_branch/das_subspace_freegen.py --out_dir runs/das_train \
    --layer 12 --k_sub 8 --merge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import read_jsonl  # noqa: E402
from src.analysis.das_span import fork_span_ids, generate_with_span_patch  # noqa: E402
from src.analysis.das_train import SubspaceU, interchange_states  # noqa: E402
from src.analysis.transition_operator import stable_seed  # noqa: E402
from src.eval.math_grade import grade  # noqa: E402


def _solve(model, tok, device, ctx_ids, args, tid, name, gt, layer=None,
           lo=None, hi=None, states=None):
    import torch
    torch.manual_seed(stable_seed(tid + name, args.seed))
    conts = generate_with_span_patch(model, tok, device, ctx_ids, args.k_rollouts,
                                     args.temperature, args.top_p, args.max_new_tokens,
                                     layer=layer, lo=lo, hi=hi, states=states)
    g = [bool(grade(c, gt)["correct"]) for c in conts]
    return sum(g) / len(g)


def run(args, model, tok, device, rows) -> None:
    import torch

    cache = torch.load(Path(args.out_dir) / f"cache_L{args.layer}.pt",
                       weights_only=False)
    by_id = {it["trace_id"]: it for it in cache}
    traces = {t["trace_id"]: t for t in read_jsonl(Path(args.run_dir) / "traces_forks.jsonl")}
    val = [it for it in cache if it["split"] != "train"]
    if args.max_eval:
        val = val[:args.max_eval]
    val = [v for i, v in enumerate(val) if i % args.num_shards == args.shard_id]

    d = cache[0]["base"].shape[1]
    u = SubspaceU(d, args.k_sub, seed=args.seed).to(device)
    u.load_state_dict(torch.load(Path(args.out_dir) /
                                 f"U_L{args.layer}_k{args.k_sub}_s{args.seed}.pt",
                                 weights_only=True))
    Q = u().detach()
    Qr = SubspaceU(d, args.k_sub, seed=args.seed + 1000).to(device)().detach()

    t0 = time.perf_counter()
    for ji, it in enumerate(val):
        tid = it["trace_id"]
        tr = traces.get(tid)
        if tr is None or not tr.get("gt_answer"):
            continue
        gt = tr["gt_answer"]
        wids, wlo, whi = fork_span_ids(tok, tr, "wrong")
        cids, _, _ = fork_span_ids(tok, tr, "correct")
        base = it["base"].to(device).float()
        donor = it["donor"].to(device).float()
        lo, hi = it["inject"]
        das_states = interchange_states(base, donor, Q)
        rnd_states = interchange_states(base, donor, Qr)
        rec = {"trace_id": tid, "layer": args.layer, "k_sub": args.k_sub,
               "seed": args.seed,
               "wrong": _solve(model, tok, device, wids, args, tid, "wrong", gt),
               "correct": _solve(model, tok, device, cids, args, tid, "correct", gt),
               "oracle": _solve(model, tok, device, wids, args, tid, "oracle", gt,
                                args.layer, lo, hi, donor),
               "das": _solve(model, tok, device, wids, args, tid, "das", gt,
                             args.layer, lo, hi, das_states),
               "random": _solve(model, tok, device, wids, args, tid, "random", gt,
                                args.layer, lo, hi, rnd_states)}
        rows.append(rec)
        if (ji + 1) % 10 == 0:
            print(f"[fg-sub shard {args.shard_id}] {ji + 1}/{len(val)} "
                  f"({(time.perf_counter() - t0) / (ji + 1):.1f}s/fork)", flush=True)


def merge(args) -> None:
    import numpy as np
    import pandas as pd
    from scipy import stats

    d = Path(args.out_dir)
    shards = sorted(d.glob(f"fgsub_L{args.layer}_k{args.k_sub}_shard*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    df.to_parquet(d / f"fgsub_L{args.layer}_k{args.k_sub}.parquet")
    gap = (df.correct - df.wrong)
    denom = float(gap.mean())

    def rec(col):
        delta = (df[col] - df.wrong)
        p_vs_rand = float(stats.wilcoxon(df[col], df.random, alternative="greater").pvalue) \
            if (df[col] - df.random).abs().gt(0).sum() >= 10 else float("nan")
        return {"mean_solve": float(df[col].mean()),
                "mean_minus_wrong": float(delta.mean()),
                "recovery": float(delta.mean() / denom) if denom != 0 else float("nan"),
                "p_gt_random": p_vs_rand}
    out = {"layer": args.layer, "k_sub": args.k_sub, "n": int(len(df)),
           "mean_solve_wrong": float(df.wrong.mean()),
           "mean_solve_correct": float(df.correct.mean()),
           "solve_gap": denom,
           "oracle": rec("oracle"), "das": rec("das"), "random": rec("random")}
    (d / f"gates_fgsub_L{args.layer}_k{args.k_sub}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/causal_graph"))
    ap.add_argument("--out_dir", type=Path, default=Path("runs/das_train"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--k_sub", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_eval", type=int, default=160)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
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

    rows = []
    run(args, model, tok, device, rows)
    pd.DataFrame(rows).to_parquet(
        args.out_dir / f"fgsub_L{args.layer}_k{args.k_sub}_shard{args.shard_id}.parquet")
    print(f"[fg-sub shard {args.shard_id}] rows {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
