#!/usr/bin/env python3
"""Label the first wrong step by rollout, and certify that labeller on humans.

The judge bake-off (job 433640) put three local judges at F1_PB 0.42-0.44 on 400
human-labelled ProcessBench traces, with Acc_error 0.29: they locate the actual
first error in under a third of the error traces, while the representation they
would be used to evaluate scores 0.566 on the same metric. Labels noisier than
the thing being measured are not a foundation for a rank claim.

This is the other standard way to get step labels, and it needs no judge. From
each prefix of a solution, sample K continuations from the model and grade them
against the known answer. The solve rate as a function of prefix length is a
value curve, and the first step after which the model can no longer recover is
the first error. It is Math-Shepherd's hard estimation, it is on-policy by
construction, and its only supervision is the final-answer grader.

What it measures is not identical to what a human annotator marks. A human marks
a step that is *wrong*; a rollout marks a step after which the model cannot
*recover*. Those come apart in both directions: a wrong step the model routinely
corrects later, and a correct-but-hard step the model cannot follow up on. So
the labeller is certified, on human annotations, before its labels are used.

**The certification is a paired test on PRM800K forks**, not a localisation test
on constructed traces. A fork is one prefix with two continuations, one rated +1
and one rated -1 by an annotator. The rollout value is computed after each. The
question is whether the value is lower after the step humans called wrong, on
the same prefix, in the same problem, which is exactly the discrimination the
labeller needs and is immune to the "errors come last" artifact that makes
PRM800K useless for whole-trace localisation.

The free calibration is run too: on trajectories that reached the *correct*
answer the rule should almost never fire, and how often it does is the labeller's
false-alarm rate on the on-policy distribution, with no human labels involved.

Reuses the rollout engine and statistics already built and tested for
cot_causal_graph_v0 and progress_usefulness_v0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.causal_graph.cg_stage2_fg import generate_batch  # noqa: E402
from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl, write_jsonl  # noqa: E402
from scripts.generate_onpolicy_steps import build_prompt, split_into_steps  # noqa: E402
from src.analysis.causal_graph import wilson_ci  # noqa: E402
from src.eval.math_grade import grade  # noqa: E402

NO_ERROR = -1


# ---------------------------------------------------------------------------
# Pure: contexts and the labelling rule
# ---------------------------------------------------------------------------

def prefix_contexts(problem: str, steps: list[str]) -> list[str]:
    """Contexts to roll out from: the bare problem, then after each step.

    Index 0 is the model's own starting point, which is what makes the curve
    readable: without it a low solve rate after step 0 could just mean the
    problem is hard.
    """
    out = [build_prompt(problem)]
    for k in range(len(steps)):
        out.append(build_prompt(problem) + "\n\n".join(steps[:k + 1]) + "\n\n")
    return out


def first_error_from_curve(curve: list[float], rule: str = "zero",
                           min_drop: float = 0.5) -> int:
    """The first step the model cannot recover from.

    `curve[0]` is the solve rate before any step, `curve[k+1]` after step k.

    zero: the first step after which no rollout reaches the answer. Math-Shepherd's
        hard estimation, and the conservative reading: it fires only on total
        collapse.
    drop: the first step whose solve rate falls by `min_drop`. Catches a step that
        wrecks the solution without making it impossible, at the cost of a
        threshold that has to be justified.
    """
    if len(curve) < 2:
        return NO_ERROR
    if rule == "zero":
        for k in range(1, len(curve)):
            if curve[k] == 0.0:
                return k - 1
        return NO_ERROR
    if rule == "drop":
        for k in range(1, len(curve)):
            if curve[k - 1] - curve[k] >= min_drop:
                return k - 1
        return NO_ERROR
    raise ValueError(f"unknown rule {rule!r}")


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(np.int64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    ss = np.asarray(s)[order]
    i = 0
    while i < len(ss):
        j = i + 1
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def solve_rates(contexts: list[str], gold: str, model, tok, device, args
                ) -> list[float]:
    """Solve rate at each context, batched over contexts."""
    rates: list[float] = []
    for i in range(0, len(contexts), args.contexts_per_batch):
        chunk = contexts[i:i + args.contexts_per_batch]
        outs = generate_batch(model, tok, device, chunk, args.k_rollouts, args)
        for gens in outs:
            hits = sum(1 for g in gens if grade(g, gold)["correct"])
            rates.append(hits / max(1, len(gens)))
    return rates


def label_trajectories(trajs: list[dict], model, tok, device, args) -> list[dict]:
    rows = []
    t0 = time.perf_counter()
    for i, tr in enumerate(trajs):
        steps = split_into_steps(tr["solution"])
        ctx = prefix_contexts(tr["problem"], steps)
        rates = solve_rates(ctx, tr["gold"], model, tok, device, args)
        fe = first_error_from_curve(rates, args.rule, args.min_drop)
        k = args.k_rollouts
        rows.append({
            "traj_uid": tr["traj_uid"], "id": tr["traj_uid"],
            "first_error": int(fe), "parse_ok": True,
            "traj_correct": bool(tr["correct"]), "n_steps": len(steps),
            "solve_curve": [float(r) for r in rates],
            "base_rate": float(rates[0]) if rates else float("nan"),
            "base_ci": list(wilson_ci(int(round(rates[0] * k)), k)) if rates else None,
            "rule": args.rule, "k_rollouts": k,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(trajs):
            fired = sum(1 for r in rows if r["first_error"] != NO_ERROR)
            print(f"[rollout] {i+1}/{len(trajs)} ({time.perf_counter()-t0:.0f}s) "
                  f"rule fired on {fired}/{len(rows)}", flush=True)
    return rows


def certify_forks(pairs: list[dict], model, tok, device, args) -> dict:
    """Paired test: is the value lower after the step humans rated -1?

    Both steps of a pair share a prefix and a problem, so nothing but the step
    differs. Reported as a win rate over pairs and as an AUROC over the pooled
    values, which is the same statistic the probes are scored with.
    """
    rows, t0 = [], time.perf_counter()
    for i, pr in enumerate(pairs):
        base = build_prompt(pr["problem"]) + (pr["prefix"] + "\n\n" if pr["prefix"] else "")
        ctx = [base + pr["positive_step"] + "\n\n", base + pr["negative_step"] + "\n\n"]
        rates = solve_rates(ctx, pr["ground_truth_answer"], model, tok, device, args)
        rows.append({"fork_id": pr["fork_id"], "v_pos": rates[0], "v_neg": rates[1]})
        if (i + 1) % 10 == 0 or i + 1 == len(pairs):
            print(f"[certify] {i+1}/{len(pairs)} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    vp = np.array([r["v_pos"] for r in rows])
    vn = np.array([r["v_neg"] for r in rows])
    decided = vp != vn
    y = np.concatenate([np.zeros(len(vp)), np.ones(len(vn))])      # 1 = human said wrong
    s = np.concatenate([-vp, -vn])                                  # lower value = more suspicious
    return {
        "n_pairs": len(rows),
        "win_rate": float((vn < vp).mean()),
        "win_rate_among_decided": float((vn[decided] < vp[decided]).mean())
        if decided.any() else float("nan"),
        "ties": float((~decided).mean()),
        "auroc": auroc(y, s),
        "mean_value_positive": float(vp.mean()), "mean_value_negative": float(vn.mean()),
        "rows": rows,
    }


def load_fork_pairs(items_path: Path, pairs_path: Path | None, limit: int) -> list[dict]:
    """(prefix, +1 step, -1 step) triples from a fork file.

    Two schemas are in the project and both are read here rather than converted,
    because a conversion step is somewhere else for a stale copy to hide.

      paired   one row per fork: question, prefix_steps, correct, wrong, gt_answer
               (the transition_operator set, 5,000 forks)
      flat     two rows per fork sharing fork_id, each with candidate_step and a
               rating of +1 or -1 (the prestudy and full fork builders)
    """
    items = read_jsonl(items_path)
    if items and "wrong" in items[0] and "correct" in items[0]:
        out = [{"fork_id": str(it["fork_id"]),
                "problem": it["question"],
                "prefix": "\n\n".join(it.get("prefix_steps") or []),
                "ground_truth_answer": it["gt_answer"],
                "positive_step": it["correct"],
                "negative_step": it["wrong"]}
               for it in items
               if it.get("correct") and it.get("wrong") and it.get("gt_answer")]
        return out[:limit] if limit > 0 else out
    by_fork: dict[str, dict] = {}
    for it in items:
        fid = it.get("fork_id")
        if fid is None:
            continue
        rating = int(it.get("rating", 0))
        slot = by_fork.setdefault(fid, {"fork_id": fid, "problem": it["problem"],
                                        "prefix": it.get("prefix", ""),
                                        "ground_truth_answer": it["ground_truth_answer"]})
        if rating == 1 and "positive_step" not in slot:
            slot["positive_step"] = it["candidate_step"]
        elif rating == -1 and "negative_step" not in slot:
            slot["negative_step"] = it["candidate_step"]
    out = [v for v in by_fork.values()
           if "positive_step" in v and "negative_step" in v and v["ground_truth_answer"]]
    return out[:limit] if limit > 0 else out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", nargs="*", default=[], type=Path,
                   help="Generator *_trajectories.jsonl to label.")
    p.add_argument("--certify_forks", type=Path, default=None,
                   help="Fork items jsonl for the paired human certification.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--model_dtype", choices=["float16", "bfloat16", "float32"],
                   default="bfloat16")
    p.add_argument("--k_rollouts", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--contexts_per_batch", type=int, default=8)
    p.add_argument("--rule", choices=["zero", "drop"], default="zero")
    p.add_argument("--min_drop", type=float, default=0.5)
    p.add_argument("--max_traces", type=int, default=0)
    p.add_argument("--n_correct_audit", type=int, default=100,
                   help="Correct trajectories to roll out as a false-alarm check. "
                        "They are labelled -1 by outcome either way; this measures "
                        "how often the rule would have disagreed.")
    p.add_argument("--max_forks", type=int, default=200)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed + args.shard_idx)
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.model_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only,
        torch_dtype=dtype).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    device = next(model.parameters()).device
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.certify_forks:
        pairs = load_fork_pairs(args.certify_forks, None, args.max_forks)
        pairs = pairs[args.shard_idx::args.num_shards]
        print(f"[certify] {len(pairs)} fork pairs, K={args.k_rollouts}", flush=True)
        rep = certify_forks(pairs, model, tok, device, args)
        rows = rep.pop("rows")
        write_jsonl(args.out, rows)
        rep.update({"model": args.model_name_or_path, "k_rollouts": args.k_rollouts,
                    "temperature": args.temperature, "source": str(args.certify_forks),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "code_commit": git_commit()})
        print(f"\npairs {rep['n_pairs']}   ties {rep['ties']:.3f}")
        print(f"the human-rated wrong step has the lower value in "
              f"{rep['win_rate']:.3f} of pairs "
              f"({rep['win_rate_among_decided']:.3f} of those the rollouts separate)")
        print(f"AUROC {rep['auroc']:.3f}   mean value  +1 step {rep['mean_value_positive']:.3f}"
              f"   -1 step {rep['mean_value_negative']:.3f}")
        if args.report:
            args.report.write_text(json.dumps(rep, indent=2))
            print(f"[certify] wrote {args.report}")
        return

    trajs: list[dict] = []
    for f in args.trajectories:
        trajs.extend(read_jsonl(f))
    trajs = [t for t in trajs if t.get("gradeable")]
    wrong = [t for t in trajs if not t["correct"]]
    right = [t for t in trajs if t["correct"]]
    if args.max_traces > 0:
        wrong = wrong[:args.max_traces]
    audit = right[:args.n_correct_audit]
    # Correct trajectories are labelled -1 from the grader, so only a sample is
    # rolled out, purely to measure how often the rule would have disagreed.
    todo = wrong + audit
    todo = todo[args.shard_idx::args.num_shards]
    print(f"[rollout] shard {args.shard_idx}/{args.num_shards}: {len(todo)} "
          f"trajectories ({len(wrong)} incorrect + {len(audit)} correct audited), "
          f"K={args.k_rollouts}", flush=True)

    rows = label_trajectories(todo, model, tok, device, args)
    # The audited correct trajectories keep the grader's verdict as their label.
    for r in rows:
        if r["traj_correct"]:
            r["rule_fired_at"] = r["first_error"]
            r["first_error"] = NO_ERROR
    write_jsonl(args.out, rows)

    aud = [r for r in rows if r["traj_correct"]]
    inc = [r for r in rows if not r["traj_correct"]]
    rep = {
        "n_labelled": len(inc), "n_audited": len(aud),
        "coverage": float(np.mean([r["first_error"] != NO_ERROR for r in inc]))
        if inc else float("nan"),
        "false_alarm_rate": float(np.mean([r["rule_fired_at"] != NO_ERROR for r in aud]))
        if aud else float("nan"),
        "mean_relative_position": float(np.mean(
            [r["first_error"] / max(1, r["n_steps"] - 1) for r in inc
             if r["first_error"] != NO_ERROR])) if inc else float("nan"),
        "mean_base_rate": float(np.mean([r["base_rate"] for r in rows])) if rows else None,
        "rule": args.rule, "min_drop": args.min_drop, "k_rollouts": args.k_rollouts,
        "model": args.model_name_or_path,
        "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": git_commit(),
    }
    print(f"\nlabelled {rep['n_labelled']} incorrect trajectories, rule found a "
          f"collapse in {rep['coverage']:.3f}")
    print(f"false alarms on {rep['n_audited']} correct trajectories: "
          f"{rep['false_alarm_rate']:.3f}")
    print(f"the collapse sits {rep['mean_relative_position']:.2f} of the way through")
    print(f"mean solve rate before any step: {rep['mean_base_rate']:.3f}")
    if args.report:
        args.report.write_text(json.dumps(rep, indent=2))
        print(f"[rollout] wrote {args.report}")


if __name__ == "__main__":
    main()
