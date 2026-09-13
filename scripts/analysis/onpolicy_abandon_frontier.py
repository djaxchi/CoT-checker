#!/usr/bin/env python3
"""Accuracy per generation token when a doomed prefix is abandoned mid-write.

This is the online arm §20.9 did not run. Guided decoding spends tokens: five
candidate steps per position, 10.7x the budget, and the branching itself costs
more accuracy than the head recovers. Abandonment spends none: one trajectory at
the policy's own temperature, killed the moment its prefix looks condemned, and
the budget handed to a fresh sample.

The signal it reads is available online for free in the generation arm. Each
step's score is computed from a context holding the problem and the steps before
it (`src/onpolicy/prompts.py`), so it is prefix-causal, and the generation-style
context is the one the sampler actually ran under.

Cost is generation tokens, the same unit as §20.9's table, so the two are
directly comparable. Everything is a replay of saved trajectories: no generation,
no fitting of any representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.onpolicy_budget_frontier import (  # noqa: E402
    collect_cells, logit,
)
from scripts.analysis.onpolicy_tiebreak_baselines import (  # noqa: E402
    fitted_questions, load_pool, read_rows,
)
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from src.analysis.onpolicy_abandon import policy_outcome  # noqa: E402
from src.analysis.onpolicy_reliability import grouped_folds  # noqa: E402
from src.analysis.onpolicy_tiebreak import cluster_bootstrap  # noqa: E402
from src.eval.math_grade import normalize_answer  # noqa: E402

ANSWER_RULES = ["majority", "majority+tie"]
# Thresholds are quantiles of the per-step score distribution, so the grid means
# the same thing whatever cell or ensemble is scored.
QUANTILES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
MIN_STEPS = [1, 2]
# Deadline for abandoning: past it the candidate is committed. None means a kill
# is allowed at any depth, which is the rule that mostly kills too late to pay.
MAX_STEPS = [1, 2, 3, 4, None]
COMPLETIONS = [1, 2, 3, 4, 5, 6, 8, 10]
# The pool holds about ten samples of ~470 tokens, so ~4,700 tokens is
# everything there is to spend.
BUDGETS = [500, 1000, 1500, 2000, 3000, 4500]


def token_counts(solution: str, n_steps: int, tokenizer) -> list[int] | None:
    """Generation cost of each step, in tokens of the policy's own tokenizer."""
    steps = split_into_steps(solution)
    if len(steps) != n_steps:
        return None
    joiner = len(tokenizer("\n\n", add_special_tokens=False)["input_ids"])
    return [len(tokenizer(s, add_special_tokens=False)["input_ids"]) + joiner
            for s in steps]


def build_candidates(pool: dict, files: list[Path], tokenizer) -> dict[str, list[dict]]:
    outcomes, traj = pool["outcomes"], pool["traj"]
    scores: dict[str, list[np.ndarray]] = defaultdict(list)
    for f in files:
        for row in read_rows(f):
            scores[row["id"]].append(logit(np.array(row["scores"], dtype=float)))
    by_problem: dict[str, list[dict]] = defaultdict(list)
    dropped = 0
    for uid, mats in scores.items():
        o = outcomes.get(uid)
        if o is None:
            continue
        s = np.mean(mats, axis=0)
        tok = token_counts(traj[uid]["solution"], len(s), tokenizer)
        if tok is None:
            # The step splitter and the encoder disagreed on this trace; a cost
            # model that guesses which step is which would be worse than a gap.
            dropped += 1
            continue
        by_problem[o["problem_id"]].append({
            "uid": uid, "step_scores": s.tolist(), "step_tokens": tok,
            "answer": normalize_answer(o.get("pred")),
            "correct": bool(o["correct"]), "final_score": float(s.max()),
        })
    if dropped:
        print(f"[build] dropped {dropped} trajectories whose step split did not "
              f"match the encoder's step count", flush=True)
    return dict(by_problem)


def sweep(by_problem: dict[str, list[dict]], problems: list[str],
          taus: list[tuple[float, float]], n_orders: int, seed: int) -> dict:
    """Every (threshold, min_steps, completions) policy on shared sampling orders."""
    rng = np.random.default_rng(seed)
    orders = {p: [rng.permutation(len(by_problem[p])) for _ in range(n_orders)]
              for p in problems}
    grid: dict[tuple, dict[str, list]] = {}
    for q, tau in taus:
        for ms in MIN_STEPS:
            for xs in MAX_STEPS:
                if xs is not None and xs < ms:
                    continue
                for c in COMPLETIONS:
                    key = (q, ms, xs, c, tau)
                    acc = {a: defaultdict(list) for a in ANSWER_RULES}
                    cost: dict[str, list] = defaultdict(list)
                    kills: list[float] = []
                    for p in problems:
                        for order in orders[p]:
                            out = policy_outcome(by_problem[p], order, tau, ms, c, xs)
                            cost[p].append(out["tokens"])
                            kills.append(out["n_drawn"] - len(out["finished"]))
                            for a in ANSWER_RULES:
                                acc[a][p].append(out[a])
                    grid[key] = {
                        "tokens": float(np.mean([v for x in cost.values() for v in x])),
                        "abandoned_per_problem": float(np.mean(kills)),
                        "per_problem_cost": {p: float(np.mean(v))
                                             for p, v in cost.items()},
                        "accuracy": {a: float(np.mean([v for x in acc[a].values()
                                                       for v in x]))
                                     for a in ANSWER_RULES},
                        "per_problem": {a: {p: float(np.mean(v))
                                            for p, v in acc[a].items()}
                                        for a in ANSWER_RULES},
                    }
    return grid


def choose_and_score(grid: dict, problems: list[str], question: dict[str, str],
                     n_folds: int, seed: int, plain_only: bool) -> dict:
    """Pick the policy on training questions under a budget, score it on held out ones."""
    folds = grouped_folds([question[p] for p in problems], n_folds, seed)
    keys = [k for k in grid if (k[4] == float("inf")) == plain_only]
    out = {a: {} for a in ANSWER_RULES}
    for a in ANSWER_RULES:
        for budget in BUDGETS:
            picked, scored, costs = [], {}, {}
            for k, test in enumerate(folds):
                train = [problems[i] for j, f in enumerate(folds) if j != k for i in f]
                held = [problems[i] for i in test]
                ok = [key for key in keys
                      if np.mean([grid[key]["per_problem_cost"][p] for p in train])
                      <= budget]
                if not ok:
                    continue
                best = max(ok, key=lambda key: np.mean(
                    [grid[key]["per_problem"][a][p] for p in train]))
                picked.append(best)
                for p in held:
                    scored[p] = grid[best]["per_problem"][a][p]
                    costs[p] = grid[best]["per_problem_cost"][p]
            if not scored:
                continue
            order = sorted(scored)
            out[a][budget] = {
                "accuracy": float(np.mean([scored[p] for p in order])),
                "tokens": float(np.mean([costs[p] for p in order])),
                "policies": [f"kill above q={k[0]:.2f} between steps "
                             f"{k[1]}-{k[2] or 'end'}, completions={k[3]}"
                             for k in picked],
                "_per_problem": scored,
            }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onpolicy_root", type=Path,
                   default=ROOT / "cot-checker-results/onpolicy_v1")
    p.add_argument("--reprobe_root", type=Path,
                   default=ROOT / "cot-checker-results/reprobe_v1")
    p.add_argument("--out_dir", type=Path, default=ROOT / "results/onpolicy_abandon")
    p.add_argument("--arms", nargs="+", default=["frozen"])
    p.add_argument("--styles", nargs="+", default=["generation"],
                   help="Generation style is the one whose states the sampler "
                        "already computed, so reading it online is free.")
    p.add_argument("--cell", default=None)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Stands in for Qwen3-8B-Base, which is not cached "
                        "locally; both use the same Qwen2 BPE vocabulary, so "
                        "step token counts agree for ordinary text.")
    p.add_argument("--subset", default="clean", choices=["clean", "all"])
    p.add_argument("--n_orders", type=int, default=40)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    pool, question = load_pool(args.onpolicy_root)
    cells = collect_cells(args.onpolicy_root, args.reprobe_root, args.arms, args.styles)
    if args.cell:
        cells = [c for c in cells if c[2].parent.name == args.cell]
    if not cells:
        raise SystemExit("no score files matched")
    by_problem = build_candidates(pool, [c[2] for c in cells], tok)

    fitted = fitted_questions(
        args.reprobe_root / "reprobe_train_judge_traces.jsonl",
        sorted(args.reprobe_root.glob("labels.shard*.jsonl")), 0, 0.15)
    problems = sorted(by_problem)
    if args.subset == "clean":
        problems = [p for p in problems if question[p] not in fitted]
    every = np.concatenate([c["step_scores"] for p in problems
                            for c in by_problem[p]])
    taus = [(q, float(np.quantile(every, q)) if q <= 1 else float("inf"))
            for q in QUANTILES]
    if not any(t == float("inf") for _, t in taus):
        taus.append((1.01, float("inf")))
    mean_len = float(np.mean([sum(c["step_tokens"]) for p in problems
                              for c in by_problem[p]]))
    print(f"[pool] {len(cells)} cells, {len(problems)} problems, "
          f"{sum(len(by_problem[p]) for p in problems)} trajectories, "
          f"mean {mean_len:.0f} generation tokens each", flush=True)

    grid = sweep(by_problem, problems, taus, args.n_orders, args.seed)
    abandon = choose_and_score(grid, problems, question, args.n_folds, args.seed,
                               plain_only=False)
    plain = choose_and_score(grid, problems, question, args.n_folds, args.seed,
                             plain_only=True)
    gaps = {}
    for a in ANSWER_RULES:
        gaps[a] = {}
        for b in BUDGETS:
            if b not in abandon[a] or b not in plain[a]:
                continue
            shared = sorted(set(abandon[a][b]["_per_problem"])
                            & set(plain[a][b]["_per_problem"]))
            paired = np.array([abandon[a][b]["_per_problem"][p]
                               - plain[a][b]["_per_problem"][p] for p in shared])
            gaps[a][b] = cluster_bootstrap(paired, [question[p] for p in shared])
    for a in ANSWER_RULES:
        for b in list(abandon[a]) + []:
            abandon[a][b].pop("_per_problem", None)
        for b in list(plain[a]):
            plain[a][b].pop("_per_problem", None)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "replay of saved trajectories with prefix-causal step scores; "
                 "no generation, no representation training",
        "cells": [str(c[2].relative_to(ROOT)) for c in cells],
        "styles": args.styles, "arms": args.arms, "subset": args.subset,
        "tokenizer": args.tokenizer,
        "n_problems": len(problems), "n_orders": args.n_orders,
        "mean_tokens_per_solution": mean_len,
        "abandon": abandon, "plain": plain, "gap_vs_plain": gaps,
        "grid": {f"q={k[0]:.2f}|steps={k[1]}-{k[2] or 'end'}|completions={k[3]}":
                 {"tokens": v["tokens"], "accuracy": v["accuracy"],
                  "abandoned_per_problem": v["abandoned_per_problem"]}
                 for k, v in grid.items()},
        "source_sha256": {str(c[2].relative_to(ROOT)):
                          hashlib.sha256(c[2].read_bytes()).hexdigest()
                          for c in cells},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{'_'.join(args.arms)}_{'_'.join(args.styles)}_{args.subset}"
    (args.out_dir / f"abandon_{tag}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False, default=float))
    text = render(report, gaps)
    (args.out_dir / f"abandon_{tag}.md").write_text(text)
    print(text)
    print(f"[wrote] {args.out_dir}/abandon_{tag}.json")
    print(f"[wrote] {args.out_dir}/abandon_{tag}.md")


def render(report: dict, gaps: dict) -> str:
    out = [f"\n## Accuracy per generation token, abandoning condemned prefixes "
           f"({report['subset']} subset, n={report['n_problems']} problems, "
           f"{report['n_orders']} sampling orders, mean "
           f"{report['mean_tokens_per_solution']:.0f} tokens per solution)\n"]
    for a in ANSWER_RULES:
        out.append(f"\n### answer rule: {a}\n")
        out.append("| token budget | plain sampling | with abandonment | gain "
                   "| 95% CI | tokens actually spent | policy chosen |")
        out.append("|---|---|---|---|---|---|---|")
        for b in BUDGETS:
            if b not in report["abandon"][a] or b not in report["plain"][a]:
                continue
            A, P, g = report["abandon"][a][b], report["plain"][a][b], gaps[a][b]
            pol = max(set(A["policies"]), key=A["policies"].count)
            out.append(f"| {b} | {P['accuracy']:.4f} | {A['accuracy']:.4f} | "
                       f"{g['delta']:+.4f} | [{g['ci95'][0]:+.4f}, "
                       f"{g['ci95'][1]:+.4f}] | {A['tokens']:.0f} vs "
                       f"{P['tokens']:.0f} | {pol} |")
    out.append("\n### the deadline family, at matched completions\n")
    out.append("Killing after step 1 only, which is the family the envelope picks "
               "out. `acc/1k` is correct answers per thousand generation tokens: "
               "the column that decides whether abandonment is worth it.\n")
    out.append("| completions | kill above q | tokens | accuracy | plain tokens "
               "| plain accuracy | acc/1k | plain acc/1k |")
    out.append("|---|---|---|---|---|---|---|---|")
    grid = report["grid"]
    for c in (1, 2, 3, 5):
        plain = grid.get(f"q=1.01|steps=1-1|completions={c}")
        if plain is None:
            continue
        for q in (0.30, 0.50, 0.70, 0.90):
            v = grid.get(f"q={q:.2f}|steps=1-1|completions={c}")
            if v is None:
                continue
            a, pa = v["accuracy"]["majority+tie"], plain["accuracy"]["majority+tie"]
            out.append(f"| {c} | {q:.2f} | {v['tokens']:.0f} | {a:.4f} | "
                       f"{plain['tokens']:.0f} | {pa:.4f} | "
                       f"{1000 * a / v['tokens']:.4f} | "
                       f"{1000 * pa / plain['tokens']:.4f} |")
    out.append("\nPolicies are chosen on training questions under the budget and "
               "scored on held-out ones; the column shows the modal choice across "
               "folds. The quantile is of the per-step score distribution, above "
               "which a prefix is condemned.")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
