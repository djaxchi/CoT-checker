#!/usr/bin/env python3
"""Does the benchmark ranking predict what a verifier is actually good for?

This is T2, and it needs no step labels at all. A verifier's downstream job is to
look at ten solutions the model wrote and pick a good one, and whether it picked
a good one is settled by the arithmetic grader. So the whole evaluation runs off
two things we already have for free: the per-step scores each trained cell
produced, and whether each trajectory reached the right answer.

That matters beyond saving money. Half the field reports step-classification
metrics and half reports test-time-scaling gains, on different models, and nobody
measures both on the same ones. These nineteen verifiers differ only in which
rows of the forward pass they read and what reads them: one trainer, one
protocol, identical frozen activations, fingerprint-checked inputs. A correlation
across them isolates the metric relationship with every nuisance variable pinned,
which is a stronger instrument than RewardBench 2's correlation across published
models of different sizes and training sets.

Four simulations, all offline over stored scores, all against baselines that
cost nothing to beat on paper and are hard to beat in practice:

  best-of-N        rank the N solutions of a problem, keep one
  weighted vote    weight each solution's answer by its score, take the winner
  trajectory AUROC does the score separate solutions that succeed from those
                   that fail, pooled across problems
  within-problem   the same question asked inside one problem, which is the one
                   best-of-N actually depends on
  step AUROC       do steps of failed solutions score higher than steps of
                   successful ones, which is the label-free shadow of F1_PB

**Aggregation is a choice, so all three standard rules are reported.** A
trajectory score has to come from its step scores somehow: the worst step (the
usual PRM rule), the mean, or the last. A verifier can win under one and lose
under another, and reporting only the rule that flatters a conclusion is how this
kind of study goes wrong.

The hypothesis worth stating before looking: picking needs only *relative*
ordering within one problem, while first-error localisation needs a threshold
calibrated *across* problems. A representation can be good at one and mediocre at
the other. If the leaderboard's winner does not win here, that is evidence
F1_PB is the wrong thing to select verifiers on, and it is the most useful
finding available from this arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import math

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.encode_prm800k_hidden_states import read_jsonl  # noqa: E402
from src.eval.math_grade import is_equiv, normalize_answer  # noqa: E402

AGGREGATIONS = ("worst_step", "mean_step", "last_step")


# ---------------------------------------------------------------------------
# Pure
# ---------------------------------------------------------------------------

def aggregate(step_scores: list[float], how: str) -> float:
    """Trajectory suspicion from its step suspicions. Higher = more suspicious."""
    a = np.asarray(step_scores, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    if how == "worst_step":
        return float(a.max())
    if how == "mean_step":
        return float(a.mean())
    if how == "last_step":
        return float(a[-1])
    raise ValueError(f"unknown aggregation {how!r}")


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


def best_of_n_hits(groups: dict[str, list[dict]], how: str) -> dict[str, float]:
    """Per problem, did the kept solution reach the right answer.

    Kept per problem rather than averaged, so the comparison against
    self-consistency can be paired: the two methods are scored on the same 300
    problems, and an unpaired interval on that difference is needlessly wide.
    """
    hits = {}
    for pid, sols in groups.items():
        scored = [(aggregate(s["scores"], how), i, s) for i, s in enumerate(sols)]
        scored.sort(key=lambda t: (t[0], t[1]))
        hits[pid] = float(scored[0][2]["correct"])
    return hits


def mcnemar(a: dict[str, float], b: dict[str, float]) -> dict:
    """Paired comparison of two picking rules over the same problems.

    Only the problems where they disagree carry information, which is what makes
    this tighter than comparing two accuracies computed independently.
    """
    shared = set(a) & set(b)
    a_only = sum(1 for k in shared if a[k] > b[k])
    b_only = sum(1 for k in shared if b[k] > a[k])
    n = a_only + b_only
    if n == 0:
        return {"n_discordant": 0, "p": float("nan"), "gap": 0.0}
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(max(a_only, b_only), n + 1))
            / 2 ** n)
    return {"n_discordant": n, "a_wins": a_only, "b_wins": b_only, "p": float(p),
            "gap": float(np.mean([a[k] for k in shared]) - np.mean([b[k] for k in shared]))}


def length_baseline(groups: dict[str, list[dict]]) -> dict:
    """What a verifier that only counts steps would score.

    Incorrect solutions run longer, so a score that quietly tracks length would
    post a respectable trajectory AUROC while reading nothing about correctness.
    This is the row that says whether it did.
    """
    y = np.array([0 if s["correct"] else 1 for sols in groups.values() for s in sols])
    L = np.array([float(len(s["scores"])) for sols in groups.values() for s in sols])
    by_len = {p: [dict(s, scores=[float(len(s["scores"]))]) for s in v]
              for p, v in groups.items()}
    return {"traj_auroc": auroc(y, L),
            "best_of_n_shortest": best_of_n(by_len, "worst_step"),
            "mean_steps_correct": float(L[y == 0].mean()) if (y == 0).any() else float("nan"),
            "mean_steps_incorrect": float(L[y == 1].mean()) if (y == 1).any() else float("nan")}


def best_of_n(groups: dict[str, list[dict]], how: str) -> float:
    """Accuracy of keeping the least suspicious solution of each problem.

    Ties are broken by the order the solutions were generated, which is arbitrary
    but fixed, so two verifiers that tie everywhere score the same rather than
    differing by whichever happened to sort first.
    """
    hits = best_of_n_hits(groups, how)
    return float(np.mean(list(hits.values()))) if hits else float("nan")


def weighted_vote(groups: dict[str, list[dict]], how: str) -> float:
    """Majority vote with each solution's weight set by 1 - its suspicion."""
    hits = []
    for _pid, sols in groups.items():
        tally: dict[str, float] = defaultdict(float)
        first: dict[str, dict] = {}
        for s in sols:
            ans = normalize_answer(s.get("pred"))
            if ans is None:
                continue
            tally[ans] += max(0.0, 1.0 - aggregate(s["scores"], how))
            first.setdefault(ans, s)
        if not tally:
            hits.append(0.0)
            continue
        win = max(tally.items(), key=lambda kv: kv[1])[0]
        hits.append(float(first[win]["correct"]))
    return float(np.mean(hits)) if hits else float("nan")


def self_consistency_hits(groups: dict[str, list[dict]]) -> dict[str, float]:
    hits = {}
    for pid, sols in groups.items():
        tally: dict[str, int] = defaultdict(int)
        first: dict[str, dict] = {}
        for s in sols:
            ans = normalize_answer(s.get("pred"))
            if ans is None:
                continue
            tally[ans] += 1
            first.setdefault(ans, s)
        if not tally:
            hits[pid] = 0.0
            continue
        win = max(tally.items(), key=lambda kv: kv[1])[0]
        hits[pid] = float(first[win]["correct"])
    return hits


def self_consistency(groups: dict[str, list[dict]]) -> float:
    """Unweighted majority: the baseline any reranker has to beat."""
    hits = []
    for _pid, sols in groups.items():
        tally: dict[str, int] = defaultdict(int)
        first: dict[str, dict] = {}
        for s in sols:
            ans = normalize_answer(s.get("pred"))
            if ans is None:
                continue
            tally[ans] += 1
            first.setdefault(ans, s)
        if not tally:
            hits.append(0.0)
            continue
        win = max(tally.items(), key=lambda kv: kv[1])[0]
        hits.append(float(first[win]["correct"]))
    return float(np.mean(hits)) if hits else float("nan")


def oracle_and_chance(groups: dict[str, list[dict]]) -> tuple[float, float]:
    """The ceiling and the floor: any-of-N, and picking one at random."""
    orc = [float(any(s["correct"] for s in sols)) for sols in groups.values()]
    chance = [float(np.mean([s["correct"] for s in sols])) for sols in groups.values()]
    return float(np.mean(orc)), float(np.mean(chance))


def spearman(a, b) -> float:
    def rank(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(1, len(x) + 1)
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i + 1
            while j < len(xs) and xs[j] == xs[i]:
                j += 1
            if j - i > 1:
                r[order[i:j]] = (i + 1 + j) / 2
            i = j
        return r
    ra, rb = rank(a) - np.mean(rank(a)), rank(b) - np.mean(rank(b))
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def load_outcomes(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(path)}


def cell_solutions(scores_path: Path, outcomes: dict[str, dict]) -> dict[str, list[dict]]:
    """problem_id -> the solutions of that problem, with per-step scores.

    A trajectory the outcomes file does not know is dropped rather than guessed
    at; that only happens if the scores and the generation run came apart.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(scores_path):
        o = outcomes.get(row["id"])
        if o is None:
            continue
        groups[o["problem_id"]].append(
            {"id": row["id"], "scores": [float(x) for x in row["scores"]],
             "correct": bool(o["correct"]), "pred": o.get("pred")})
    return dict(groups)


def within_problem_auroc(groups: dict[str, list[dict]], how: str) -> float:
    """AUROC computed inside each problem, then averaged over problems.

    The pooled trajectory AUROC answers "given two solutions to *different*
    problems, does the verifier rank the failing one higher", and a score that
    only tracked problem difficulty would do well at that while being useless
    for picking. Best-of-N asks the other question: given ten solutions to the
    *same* problem, is the good one ranked first. Only problems with both a
    correct and an incorrect solution can contribute; the rest are undefined.
    """
    vals = []
    for sols in groups.values():
        y = np.array([0 if s["correct"] else 1 for s in sols])
        if y.min() == y.max():
            continue
        sc = np.array([aggregate(s["scores"], how) for s in sols])
        a = auroc(y, sc)
        if not np.isnan(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_cell(groups: dict[str, list[dict]], sc_hits: dict[str, float] | None = None
                  ) -> dict:
    out: dict = {}
    if sc_hits is not None:
        m = mcnemar(best_of_n_hits(groups, "worst_step"), sc_hits)
        out["vs_self_consistency_gap"] = m["gap"]
        out["vs_self_consistency_p"] = m["p"]
        out["vs_self_consistency_discordant"] = m["n_discordant"]
    for how in AGGREGATIONS:
        out[f"best_of_n__{how}"] = best_of_n(groups, how)
        out[f"within_problem_auroc__{how}"] = within_problem_auroc(groups, how)
        out[f"weighted_vote__{how}"] = weighted_vote(groups, how)
        y = np.array([0 if s["correct"] else 1
                      for sols in groups.values() for s in sols])
        sc = np.array([aggregate(s["scores"], how)
                       for sols in groups.values() for s in sols])
        out[f"traj_auroc__{how}"] = auroc(y, sc)
    # Step-level, label-free: steps of failed solutions against steps of
    # successful ones. The shadow of F1_PB that needs no annotation.
    ys, ss = [], []
    for sols in groups.values():
        for s in sols:
            ys.extend([0 if s["correct"] else 1] * len(s["scores"]))
            ss.extend(s["scores"])
    out["step_auroc_outcome"] = auroc(np.array(ys), np.array(ss))
    out["n_problems"] = len(groups)
    out["n_solutions"] = int(sum(len(v) for v in groups.values()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid_root", required=True, type=Path)
    p.add_argument("--scores_name", default="onpolicy_verifier",
                   help="Suffix of the per-cell pb_step_scores file.")
    p.add_argument("--outcomes", required=True, type=Path,
                   help="build_pb_traces *_outcomes.jsonl: one row per trajectory.")
    p.add_argument("--offpolicy_metric", type=Path, default=None,
                   help="JSON of {cell: F1_PB} from the off-policy leaderboard, "
                        "to correlate the downstream ranking against.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    outcomes = load_outcomes(args.outcomes)
    per_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in sorted(args.grid_root.iterdir()):
        rj, sj = d / "results.json", d / f"pb_step_scores_{args.scores_name}.jsonl"
        if not (rj.exists() and sj.exists()):
            continue
        res = json.loads(rj.read_text())
        groups = cell_solutions(sj, outcomes)
        if not groups:
            continue
        per_cell[(res["rep"], res["learner"])].append(
            evaluate_cell(groups, self_consistency_hits(groups)))
    if not per_cell:
        raise SystemExit("no cells with scores for this split")

    keys = sorted(per_cell)
    # Baselines depend only on the outcomes, not on any verifier.
    base_groups: dict[str, list[dict]] = defaultdict(list)
    for r in outcomes.values():
        base_groups[r["problem_id"]].append(
            {"id": r["id"], "scores": [0.5], "correct": bool(r["correct"]),
             "pred": r.get("pred")})
    oracle, chance = oracle_and_chance(base_groups)
    sc = self_consistency(base_groups)
    lb = length_baseline(base_groups)

    def mean_of(k, cell):
        return float(np.mean([r[k] for r in per_cell[cell]]))

    print(f"{len(keys)} cells, {len(base_groups)} problems, "
          f"{sum(len(v) for v in base_groups.values())} solutions\n")
    print(f"  pick one at random   {chance:.3f}")
    print(f"  self-consistency     {sc:.3f}      <- the baseline to beat")
    print(f"  any-of-N (oracle)    {oracle:.3f}      <- the ceiling")
    print(f"\n  length alone: trajectory AUROC {lb['traj_auroc']:.3f}, and keeping the "
          f"shortest solution scores {lb['best_of_n_shortest']:.3f}")
    print(f"  (incorrect solutions run {lb['mean_steps_incorrect']:.1f} steps against "
          f"{lb['mean_steps_correct']:.1f}, so a score that quietly tracked length "
          f"would land near that AUROC)\n")

    print(f"{'cell':<44}{'BoN worst':>11}{'vs SC':>8}{'p':>7}"
          f"{'pooled':>9}{'within':>9}{'step AUC':>10}")
    rows = []
    for k in sorted(keys, key=lambda c: -mean_of("best_of_n__worst_step", c)):
        r = {"rep": k[0], "learner": k[1], "n_seeds": len(per_cell[k]),
             **{m: mean_of(m, k) for m in per_cell[k][0] if m.startswith(
                 ("best_of_n", "weighted_vote", "traj_auroc", "step_auroc",
                  "within_problem_auroc", "vs_self_consistency"))}}
        rows.append(r)
        print(f"{k[0] + ' x ' + k[1]:<44}"
              f"{r['best_of_n__worst_step']:>11.3f}"
              f"{r.get('vs_self_consistency_gap', float('nan')):>+8.3f}"
              f"{r.get('vs_self_consistency_p', float('nan')):>7.3f}"
              f"{r['traj_auroc__worst_step']:>9.3f}"
              f"{r['within_problem_auroc__worst_step']:>9.3f}"
              f"{r['step_auroc_outcome']:>10.3f}")

    report = {"baselines": {"random": chance, "self_consistency": sc, "oracle": oracle,
                            "length_only": lb},
              "n_problems": len(base_groups), "cells": rows,
              "scores_name": args.scores_name}

    if args.offpolicy_metric and args.offpolicy_metric.exists():
        off = json.loads(args.offpolicy_metric.read_text())
        pairs = [(off[f"{r['rep']} x {r['learner']}"], r) for r in rows
                 if f"{r['rep']} x {r['learner']}" in off]
        if len(pairs) >= 4:
            x = [a for a, _ in pairs]
            print(f"\nDoes the benchmark rank predict the downstream rank? "
                  f"(n={len(pairs)} cells)")
            corr = {}
            for m in ("best_of_n__worst_step", "best_of_n__mean_step",
                      "weighted_vote__worst_step", "traj_auroc__worst_step",
                      "within_problem_auroc__worst_step", "step_auroc_outcome"):
                rho = spearman(x, [r[m] for _, r in pairs])
                corr[m] = rho
                print(f"  off-policy F1_PB vs {m:<26} Spearman {rho:+.3f}")
            report["correlation_with_offpolicy_f1pb"] = corr
            print("\n  At this many cells a Spearman carries roughly a +/-0.3 "
                  "interval, so this separates a strong relationship from none "
                  "and cannot separate 0.5 from 0.8.")

    print("\nNo step labels were used anywhere above. Every number is scored "
          "against whether the solution reached the right answer.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[downstream] wrote {args.out}")


if __name__ == "__main__":
    main()
