#!/usr/bin/env python3
"""cot_causal_graph_v0 Stage 2: free-generation rollouts (GPU).

Behavioral edge family (spec: docs/cot_causal_graph_v0_plan.md): K sampled
continuations per context, graded by final answer. Contexts:

Arm forks (at the fork point t only):
  base        question + golden steps through t
  swap_wrong / swap_xprob / swap_pos     same prefix with the fork step swapped
Arm onpolicy (model's own graded trajectories):
  prefix_i    base solve-from-here curve, every prefix length i = 0..T-1
  delete_i / swap_xtrace_i               at selected sites (default: the 2
              highest-probe steps from stage 1 + 2 random; --interv_policy all
              for every step)

The pilot block (--pilot, first N arm-F traces) reports per-site minimal
detectable deltas at this K (gate G2) before committing the full budget.

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python scripts/causal_graph/cg_stage2_fg.py \
    --run_dir runs/causal_graph --model_name_or_path Qwen/Qwen2.5-7B \
    --arm forks --shard_id 0 --num_shards 4
Merge + gates (CPU):
  python scripts/causal_graph/cg_stage2_fg.py --run_dir runs/causal_graph --merge
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl  # noqa: E402
from scripts.generate_onpolicy_steps import build_prompt, split_into_steps  # noqa: E402
from src.analysis.causal_graph import wilson_ci  # noqa: E402
from src.analysis.transition_operator import stable_seed  # noqa: E402
from src.eval.math_grade import grade  # noqa: E402


def fork_contexts(tr: dict) -> list[tuple[str, str]]:
    """(context_name, prompt_text) pairs for one arm-F trace."""
    t = tr["fork_t"]
    prefix = [tr["question"]] + tr["steps"][:t]

    def prompt(step_t: str) -> str:
        return "\n".join(prefix + [step_t]) + "\n"

    out = [("base", prompt(tr["steps"][t])),
           ("swap_wrong", prompt(tr["wrong_step"])),
           ("swap_xprob", prompt(tr["xprob_step"]))]
    if tr.get("alt_pos_step"):
        out.append(("swap_pos", prompt(tr["alt_pos_step"])))
    return out


def onpolicy_contexts(tr: dict, sites: list[int], pool, rng) -> list[tuple[str, str]]:
    """Base solve-from-here curve for every prefix + swap interventions at sites.

    No FG delete context: rolling out after deleting step i is identical to
    rolling out from prefix i-1, which the base curve already covers (the curve
    differences s_i - s_{i-1} ARE the free-generation deletion edges)."""
    from src.analysis.causal_graph import length_matched_step
    steps = split_into_steps(tr["solution"])
    head = build_prompt(tr["problem"])
    out = []
    for i in range(len(steps)):
        out.append((f"prefix_{i}", head + "\n\n".join(steps[:i + 1]) + "\n\n"))
    for i in sites:
        if i >= len(steps):
            continue
        swap = length_matched_step(rng, pool, len(steps[i].split()),
                                   exclude_key=tr["traj_uid"])
        out.append((f"swap_xtrace_{i}",
                    head + "\n\n".join(steps[:i] + [swap]) + "\n\n"))
    return out


def pick_sites(tr: dict, n_steps: int, probe_by_step: dict, rng,
               policy: str) -> list[int]:
    if policy == "all":
        return list(range(n_steps))
    scored = [(probe_by_step.get(i, float("-inf")), i) for i in range(n_steps)]
    top = [i for _, i in sorted(scored, reverse=True)[:2]]
    rest = [i for i in range(n_steps) if i not in top]
    rng.shuffle(rest)
    return sorted(top + rest[:2])


def generate_batch(model, tok, device, prompts: list[str], k: int, args):
    """Left-padded batched sampling; returns list of k-lists of continuations."""
    import torch
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        gen = model.generate(
            **enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_new_tokens, num_return_sequences=k,
            pad_token_id=tok.pad_token_id)
    width = enc["input_ids"].shape[1]
    texts = tok.batch_decode(gen[:, width:], skip_special_tokens=True)
    return [texts[i * k:(i + 1) * k] for i in range(len(prompts))]


def run_arm(args, model, tok, device, arm: str, rows: list[dict]) -> None:
    import torch
    if arm == "forks":
        traces = read_jsonl(args.run_dir / "traces_forks.jsonl")
        if args.pilot:
            traces = traces[:args.pilot_traces]
        mine = [t for i, t in enumerate(traces) if i % args.num_shards == args.shard_id]
        jobs = [(tr, fork_contexts(tr)) for tr in mine]
        meta = {tr["trace_id"]: tr for tr in mine}
    else:
        trajs = [t for t in read_jsonl(args.onpolicy_trajectories) if t.get("gradeable")]
        rng = random.Random(args.seed)
        rng.shuffle(trajs)
        cor = [t for t in trajs if t["correct"]][:args.max_onpolicy_traces // 2]
        inc = [t for t in trajs if not t["correct"]][:args.max_onpolicy_traces // 2]
        trajs = cor + inc
        probe_lookup: dict[tuple[str, int], float] = {}
        nf = args.run_dir / "stage1" / "node_features.parquet"
        if nf.exists() and args.interv_policy != "all":
            import pandas as pd
            df = pd.read_parquet(nf)
            for r in df[df.arm == "onpolicy"].itertuples():
                probe_lookup[(r.trace_id, r.step_idx)] = r.probe_l28
        pool = [(t["traj_uid"], s) for t in trajs for s in split_into_steps(t["solution"])]
        mine = [t for i, t in enumerate(trajs) if i % args.num_shards == args.shard_id]
        jobs, meta = [], {}
        for tr in mine:
            steps = split_into_steps(tr["solution"])
            if len(steps) < 3:
                continue
            r = random.Random(stable_seed(tr["traj_uid"], args.seed + 2))
            sites = pick_sites(tr, len(steps),
                               {i: probe_lookup.get((tr["traj_uid"], i), float("-inf"))
                                for i in range(len(steps))},
                               r, args.interv_policy)
            jobs.append((tr, onpolicy_contexts(tr, sites, pool, r)))
            meta[tr["traj_uid"]] = tr

    t0 = time.perf_counter()
    for ji, (tr, ctxs) in enumerate(jobs):
        tid = tr.get("trace_id") or tr["traj_uid"]
        gt = tr.get("gt_answer") or tr["gold"]
        for lo in range(0, len(ctxs), args.contexts_per_batch):
            chunk = ctxs[lo:lo + args.contexts_per_batch]
            torch.manual_seed(stable_seed(tid + chunk[0][0], args.seed))
            outs = generate_batch(model, tok, device, [p for _, p in chunk],
                                  args.k_rollouts, args)
            for (name, _), conts in zip(chunk, outs):
                for ri, cont in enumerate(conts):
                    g = grade(cont, gt)
                    rows.append({
                        "arm": arm, "trace_id": tid,
                        "split": tr.get("split", "test"), "context": name,
                        "t": tr.get("fork_t") if arm == "forks"
                             else (int(name.split("_")[-1]) if "_" in name else -1),
                        "rollout_idx": ri, "correct": bool(g["correct"]),
                        "gradeable": bool(g["gradeable"]),
                        "pred": (g["pred"] or "")[:60],
                        "traj_correct": tr.get("correct"),
                    })
        if (ji + 1) % 5 == 0:
            el = time.perf_counter() - t0
            print(f"[fg {arm} shard {args.shard_id}] {ji + 1}/{len(jobs)} "
                  f"({el / (ji + 1):.1f}s/trace)", flush=True)


def merge(args) -> None:
    import pandas as pd
    from scipy import stats
    d = args.run_dir / "stage2"
    shard_files = [p for p in sorted(d.glob("fg_rollouts_shard*.parquet"))
                   if "_pilot" not in p.name]
    roll = pd.concat([pd.read_parquet(p) for p in shard_files], ignore_index=True)
    roll.to_parquet(d / "fg_rollouts.parquet")
    cur = (roll.groupby(["arm", "trace_id", "split", "context", "t"])
           .agg(n=("correct", "size"), k=("correct", "sum"),
                gradeable_rate=("gradeable", "mean")).reset_index())
    cur["solve_rate"] = cur.k / cur.n
    ci = cur.apply(lambda r: wilson_ci(int(r.k), int(r.n)), axis=1)
    cur["ci_lo"] = [c[0] for c in ci]
    cur["ci_hi"] = [c[1] for c in ci]
    cur.to_parquet(d / "fg_curves.parquet")

    gates: dict = {"gradeable_rate_overall": float(roll.gradeable.mean())}
    fk = cur[cur.arm == "forks"].pivot_table(index="trace_id", columns="context",
                                             values="solve_rate", aggfunc="first")
    if "swap_wrong" in fk and "base" in fk:
        both = fk[["base", "swap_wrong"]].dropna()
        delta = (both.swap_wrong - both.base)
        gates["fg_wrong_effect"] = {
            "n": int(len(both)), "mean_delta": float(delta.mean()),
            "median_base": float(both.base.median()),
            "median_wrong": float(both.swap_wrong.median()),
            "p_wilcoxon_less": float(stats.wilcoxon(
                both.swap_wrong, both.base, alternative="less").pvalue)
            if len(both) >= 20 and (delta != 0).any() else float("nan")}
        k = int(roll[roll.arm == "forks"].groupby("context").rollout_idx.max().max()) + 1
        # minimal per-site detectable drop at this K: Wilson CI half-width at p=0.5
        lo, hi = wilson_ci(k // 2, k)
        gates["G2_fg_power"] = {
            "k_rollouts": k,
            "per_site_ci_halfwidth_p05": float((hi - lo) / 2),
            "pass_per_site_large_effects_only": bool((hi - lo) / 2 <= 0.35)}
    for ctrl in ("swap_xprob", "swap_pos"):
        if ctrl in fk and "swap_wrong" in fk and "base" in fk:
            both = fk[["base", "swap_wrong", ctrl]].dropna()
            if len(both) >= 20:
                dw = (both.swap_wrong - both.base).abs().to_numpy()
                dc = (both[ctrl] - both.base).abs().to_numpy()
                u = stats.mannwhitneyu(dw, dc, alternative="greater")
                gates[f"G3_fg_wrong_vs_{ctrl}"] = {
                    "n": int(len(both)),
                    "auc": float(u.statistic / (len(dw) * len(dc))),
                    "p": float(u.pvalue)}
    (d / "gates_stage2.json").write_text(json.dumps(gates, indent=2))
    print(json.dumps(gates, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/causal_graph"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--arm", choices=["forks", "onpolicy"], default="forks")
    ap.add_argument("--onpolicy_trajectories", type=Path,
                    default=Path("runs/causal_graph/onpolicy_trajectories.jsonl"))
    ap.add_argument("--max_onpolicy_traces", type=int, default=200)
    ap.add_argument("--interv_policy", choices=["probe_top2_rand2", "all"],
                    default="probe_top2_rand2")
    ap.add_argument("--k_rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--contexts_per_batch", type=int, default=4)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot_traces", type=int, default=50)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out_dir = args.run_dir / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args)
        return

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only).to(device).eval()

    rows: list[dict] = []
    run_arm(args, model, tok, device, args.arm, rows)
    tag = "_pilot" if args.pilot else ""
    pd.DataFrame(rows).to_parquet(
        out_dir / f"fg_rollouts_shard{args.shard_id}{tag}.parquet")
    (out_dir / f"manifest_shard{args.shard_id}{tag}.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_rows": len(rows)}, indent=2))
    print(f"[stage2 shard {args.shard_id}] rows {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
