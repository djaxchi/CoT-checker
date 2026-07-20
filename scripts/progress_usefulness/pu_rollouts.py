#!/usr/bin/env python3
"""progress_usefulness_v0 P1: causal-label gate via free-generation rollouts (GPU).

Turns the +1/0 ANNOTATION prior into a causal utility estimate. For each
same-prefix pair (progress s+, neutral s0) sharing prefix p, sample K free
generations from three contexts and grade each by final answer:

  base       build_prompt(problem) + p                 -> solve rate s(p)
  progress   build_prompt(problem) + p + s+            -> solve rate s(p, s+)
  neutral    build_prompt(problem) + p + s0            -> solve rate s(p, s0)

Utility U(s|p) = solve_rate(p, s) - solve_rate(p). The causal-label gate asks
whether progress candidates actually raise Qwen's solve-from-here probability
more than neutral candidates: mean U(+1) > mean U(0), and per-fork
U(+1) > U(0). Confirmed pairs (U_prog > U_neu) are the clean unit P2 encodes.

Reuses the cot_causal_graph_v0 rollout engine (generate_batch, grade, wilson_ci,
stable_seed) and the on-policy prompt (build_prompt).

Shard across the 4 GPUs of one node with CUDA_VISIBLE_DEVICES + --shard_id
(see feedback: TamIA H100s allocate by whole node; shard in-node, not by array).

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python scripts/progress_usefulness/pu_rollouts.py \
    --pairs_dir data/pu --out_dir runs/pu/rollouts \
    --model_name_or_path Qwen/Qwen2.5-7B --local_files_only \
    --shard_id 0 --num_shards 4
Merge + gate (CPU):
  python scripts/progress_usefulness/pu_rollouts.py --out_dir runs/pu/rollouts --merge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.causal_graph.cg_stage2_fg import generate_batch  # noqa: E402
from scripts.encode_prm800k_hidden_states import (  # noqa: E402
    git_commit,
    read_jsonl,
    write_jsonl,
)
from scripts.generate_onpolicy_steps import build_prompt  # noqa: E402
from src.analysis.transition_operator import stable_seed  # noqa: E402
from src.eval.math_grade import grade  # noqa: E402

CONTEXTS = ("base", "progress", "neutral")


# --------------------------------------------------------------------------- #
# Context assembly (pure)
# --------------------------------------------------------------------------- #

def pair_contexts(prog_item: dict, neu_item: dict) -> tuple[str, list[tuple[str, str]]]:
    """Return (gold_answer, [(context_name, prompt_text), ...]) for one pair.

    ``base`` is the shared prefix with no candidate; ``progress``/``neutral`` append
    the respective candidate step. Prefix/step joined on the PRM800K \\n\\n convention.
    """
    problem = prog_item["problem"]
    prefix = prog_item["prefix"]
    gt = prog_item.get("ground_truth_answer", "")
    head = build_prompt(problem)

    def ctx(step: str) -> str:
        base = head + (prefix + "\n\n" if prefix else "")
        return base + (step + "\n\n" if step else "")

    return gt, [
        ("base", ctx("")),
        ("progress", ctx(prog_item["candidate_step"])),
        ("neutral", ctx(neu_item["candidate_step"])),
    ]


# --------------------------------------------------------------------------- #
# Aggregation + gate (pure, no pandas — unit-testable)
# --------------------------------------------------------------------------- #

def solve_rates(rows: list[dict]) -> dict[str, dict[str, tuple[int, int]]]:
    """fork_id -> context -> (k_correct, n_rollouts)."""
    agg: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        cell = agg[r["fork_id"]][r["context"]]
        cell[0] += 1 if r["correct"] else 0
        cell[1] += 1
    return {f: {c: (k, n) for c, (k, n) in ctxs.items()} for f, ctxs in agg.items()}


def pair_utilities(rows: list[dict]) -> list[dict]:
    """Per-fork solve rates and utilities for forks with all three contexts."""
    sr = solve_rates(rows)
    out: list[dict] = []
    for fork_id, ctxs in sr.items():
        if not set(CONTEXTS) <= set(ctxs):
            continue
        (kb, nb), (kp, npr), (kn, nn) = ctxs["base"], ctxs["progress"], ctxs["neutral"]
        sb, sp, sn = kb / nb, kp / npr, kn / nn
        out.append({
            "fork_id": fork_id,
            "n_base": nb, "n_progress": npr, "n_neutral": nn,
            "sr_base": sb, "sr_progress": sp, "sr_neutral": sn,
            "U_progress": sp - sb, "U_neutral": sn - sb,
        })
    return out


def gate_summary(utils: list[dict]) -> dict:
    """Causal-label gate statistics over per-fork utilities."""
    n = len(utils)
    if n == 0:
        return {"n_forks": 0}
    up = [u["U_progress"] for u in utils]
    un = [u["U_neutral"] for u in utils]
    sp = [u["sr_progress"] for u in utils]
    sn = [u["sr_neutral"] for u in utils]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    frac_prog_gt = sum(1 for u in utils if u["U_progress"] > u["U_neutral"]) / n
    summ = {
        "n_forks": n,
        "mean_U_progress": mean(up),
        "mean_U_neutral": mean(un),
        "mean_solve_progress": mean(sp),
        "mean_solve_neutral": mean(sn),
        "frac_forks_Uprog_gt_Uneu": frac_prog_gt,
        "n_confirmed_Uprog_gt_Uneu": sum(1 for u in utils if u["U_progress"] > u["U_neutral"]),
    }
    try:
        from scipy import stats
        diffs = [a - b for a, b in zip(up, un)]
        if n >= 20 and any(d != 0 for d in diffs):
            summ["wilcoxon_Uprog_gt_Uneu_p"] = float(
                stats.wilcoxon(up, un, alternative="greater").pvalue)
    except Exception:
        pass
    return summ


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_pairs(pairs_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """Return (pairs, items_by_uid). Pairs from train+val, tagged with split."""
    items_by_uid: dict[str, dict] = {}
    for name in ("pu_train_items.jsonl", "pu_val_items.jsonl"):
        for it in read_jsonl(pairs_dir / name):
            items_by_uid[it["item_uid"]] = it
    pairs: list[dict] = []
    for name, split in (("pu_train_pairs.jsonl", "train"), ("pu_val_pairs.jsonl", "val")):
        for pr in read_jsonl(pairs_dir / name):
            pr = dict(pr)
            pr["split"] = split
            pairs.append(pr)
    return pairs, items_by_uid


# --------------------------------------------------------------------------- #
# Rollout run
# --------------------------------------------------------------------------- #

def run_shard(args, model, tok, device, pairs, items_by_uid) -> list[dict]:
    import torch
    mine = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard_id]
    if args.limit:
        mine = mine[: args.limit]
    rows: list[dict] = []
    t0 = time.perf_counter()
    for ji, pr in enumerate(mine):
        prog = items_by_uid[pr["progress_uid"]]
        neu = items_by_uid[pr["neutral_uid"]]
        gt, ctxs = pair_contexts(prog, neu)
        fid = pr["fork_id"]
        for lo in range(0, len(ctxs), args.contexts_per_batch):
            chunk = ctxs[lo:lo + args.contexts_per_batch]
            torch.manual_seed(stable_seed(fid + chunk[0][0], args.seed))
            outs = generate_batch(model, tok, device, [p for _, p in chunk],
                                  args.k_rollouts, args)
            for (name, _), conts in zip(chunk, outs):
                for ri, cont in enumerate(conts):
                    g = grade(cont, gt)
                    rows.append({
                        "fork_id": fid, "split": pr["split"], "context": name,
                        "rollout_idx": ri, "correct": bool(g["correct"]),
                        "gradeable": bool(g["gradeable"]), "pred": (g["pred"] or "")[:60],
                    })
        if (ji + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"[pu-fg shard {args.shard_id}] {ji + 1}/{len(mine)} "
                  f"({el / (ji + 1):.1f}s/pair)", flush=True)
    return rows


def merge(args) -> None:
    import pandas as pd
    shard_files = sorted(args.out_dir.glob("pu_rollouts_shard*.parquet"))
    if not shard_files:
        sys.exit(f"no shard parquet files in {args.out_dir}")
    roll = pd.concat([pd.read_parquet(p) for p in shard_files], ignore_index=True)
    roll.to_parquet(args.out_dir / "pu_rollouts.parquet")

    utils = pair_utilities(roll.to_dict("records"))
    # split-aware summaries
    split_of = dict(zip(roll.fork_id, roll.split))
    for u in utils:
        u["split"] = split_of.get(u["fork_id"], "?")
    pd.DataFrame(utils).to_parquet(args.out_dir / "pu_fork_utilities.parquet")

    gates = {
        "gradeable_rate_overall": float(roll.gradeable.mean()),
        "all": gate_summary(utils),
        "train": gate_summary([u for u in utils if u["split"] == "train"]),
        "val": gate_summary([u for u in utils if u["split"] == "val"]),
    }
    # confirmed pairs for P2 (progress raises solve-from-here more than neutral)
    confirmed = [{"fork_id": u["fork_id"], "split": u["split"],
                  "U_progress": u["U_progress"], "U_neutral": u["U_neutral"]}
                 for u in utils if u["U_progress"] > u["U_neutral"]]
    write_jsonl(args.out_dir / "pu_confirmed_forks.jsonl", confirmed)
    (args.out_dir / "pu_gates.json").write_text(json.dumps(gates, indent=2))
    print(json.dumps(gates, indent=2))
    print(f"[pu-fg merge] confirmed forks: {len(confirmed)}/{len(utils)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs_dir", type=Path, help="dir with pu_{train,val}_{items,pairs}.jsonl")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--k_rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--contexts_per_batch", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="cap pairs per shard (pilot)")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args)
        return
    if args.pairs_dir is None:
        ap.error("--pairs_dir required unless --merge")

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only).to(device).eval()

    pairs, items_by_uid = load_pairs(args.pairs_dir)
    print(f"[pu-fg shard {args.shard_id}/{args.num_shards}] pairs total={len(pairs)}",
          flush=True)
    rows = run_shard(args, model, tok, device, pairs, items_by_uid)
    pd.DataFrame(rows).to_parquet(args.out_dir / f"pu_rollouts_shard{args.shard_id}.parquet")
    (args.out_dir / f"manifest_shard{args.shard_id}.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_rows": len(rows)}, indent=2))
    print(f"[pu-fg shard {args.shard_id}] rows {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
