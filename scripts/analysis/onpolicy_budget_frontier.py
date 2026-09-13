#!/usr/bin/env python3
"""What is a verifier worth when you cannot afford ten samples?

Every downstream comparison in §20 was run at N=10, where self-consistency is
strong and the verifier loses to it. That is one point on a curve. At N=1 a vote
does not exist, at N=2 it is a coin flip, and a verifier is the only thing in the
room that can tell a good trajectory from a bad one. This traces the whole
accuracy-versus-samples curve and asks where, if anywhere, the verifier pays.

The stopping rule and the answer rule are varied independently, because a gain
from resolving tied votes (§20.12) is not a gain from spending samples well and
the two must not be added up by accident:

  stopping   fixed budget | wait for a vote margin | a fitted rule reading
             vote features, verifier features, or both
  answer     plain majority with ties as coin flips | majority with the
             verifier resolving ties

Stopping models are fitted on training folds and every threshold is chosen on
training folds under the budget constraint, so a reported operating point was
never selected on the problems it is scored on. Folds hold out whole question
texts. CPU only; the candidates and their scores are read from disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from typing import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.onpolicy_tiebreak_baselines import (  # noqa: E402
    fitted_questions, load_pool, read_rows,
)
from src.analysis.onpolicy_budget import (  # noqa: E402
    ANSWER_CHEAP_TIE, ANSWER_MAJORITY, ANSWER_VERIFIER_TIE, accuracy_at_budget,
    frontier_envelope, simulate,
)
from src.analysis.onpolicy_reliability import grouped_folds  # noqa: E402
from src.analysis.onpolicy_tiebreak import cluster_bootstrap  # noqa: E402
from src.eval.math_grade import normalize_answer  # noqa: E402

ANSWER_RULES = [ANSWER_MAJORITY, ANSWER_CHEAP_TIE, ANSWER_VERIFIER_TIE]
STOP_MODELS = ["vote", "score", "both"]
TAUS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.7, 0.8, 0.9, 1.01]
BUDGETS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load_scores(files: list[Path], uids: list[str]) -> np.ndarray:
    """Worst-step suspicion per candidate, per cell, in log-odds.

    Averaging probabilities that saturate at 1.0 throws away the ordering among
    the confident ones, so the average is taken in log-odds. A single cell is
    just the one-column case of this.
    """
    pos = {u: i for i, u in enumerate(uids)}
    out = np.full((len(files), len(uids)), np.nan)
    for k, f in enumerate(files):
        for row in read_rows(f):
            i = pos.get(row["id"])
            if i is not None:
                out[k, i] = max(row["scores"])
    if np.isnan(out).any():
        raise SystemExit("a cell does not score every retained trajectory")
    return logit(out)


def collect_cells(onpolicy_root: Path, reprobe_root: Path, arms: list[str],
                  styles: list[str]) -> list[tuple[str, str, Path]]:
    found = []
    for arm, root in (("frozen", onpolicy_root / "grid"),
                      ("retrained", reprobe_root / "grid")):
        if arm not in arms:
            continue
        for style in styles:
            found += [(arm, style, f) for f in
                      sorted(root.rglob(f"*scores*{style}*.jsonl"))]
    return found


def margin_diagnostics(answers, correct, quality, members, problems) -> dict:
    """Who is informative where, at the full pool.

    The vote's margin splits the problems into a regime where counting already
    settles the answer and one where it cannot. Ranking quality is measured
    within a problem, since that is the comparison best-of-N actually makes.
    """
    from sklearn.metrics import roc_auc_score
    buckets: dict[str, dict[str, list]] = defaultdict(
        lambda: {"majority": [], "oracle": [], "verifier": [], "agreement": []})
    for p in problems:
        idx = members[p]
        counts = defaultdict(int)
        for i in idx:
            if answers[i] is not None:
                counts[answers[i]] += 1
        if not counts:
            continue
        order = sorted(counts.values(), reverse=True)
        margin = order[0] - (order[1] if len(order) > 1 else 0)
        name = "0-1" if margin <= 1 else "2-3" if margin <= 3 else "4+"
        top = [a for a, n in counts.items() if n == order[0]]
        b = buckets[name]
        b["majority"].append(float(np.mean(
            [correct[next(i for i in idx if answers[i] == a)] for a in top])))
        b["oracle"].append(float(any(correct[i] for i in idx)))
        y = [int(bool(correct[i])) for i in idx]
        if len(set(y)) == 2:
            # Within-problem z, so a problem's overall difficulty cannot leak in.
            q = np.array([quality[i] for i in idx], dtype=float)
            b["verifier"].append(roc_auc_score(y, (q - q.mean()) / (q.std() + 1e-9)))
            b["agreement"].append(roc_auc_score(
                y, [counts.get(answers[i], 0) for i in idx]))
    return {"by_margin": {k: {"n": len(v["majority"]),
                              "majority": float(np.mean(v["majority"])),
                              "oracle": float(np.mean(v["oracle"])),
                              "within_auc_verifier": mean_or_none(np.array(v["verifier"])),
                              "within_auc_agreement": mean_or_none(np.array(v["agreement"]))}
                          for k, v in sorted(buckets.items())}}


def mean_or_none(x: np.ndarray) -> float | None:
    return float(x.mean()) if x.size else None


def flatten(runs: dict[str, list], problems: list[str]) -> dict:
    """One row per (problem, sampling order, draw), so a fold predicts once.

    Predicting per run turned out to cost minutes of sklearn call overhead for
    seconds of arithmetic.
    """
    vote, score, y_maj, y_tie, y_cheap, slices = [], [], [], [], [], []
    tied, lazy = [], []
    for pi, p in enumerate(problems):
        for run in runs[p]:
            tj = [st for st in run if st is not None]
            if not tj:
                continue
            start = len(vote)
            for st in tj:
                vote.append(st["vote"])
                score.append(st["score"])
                y_maj.append(st[ANSWER_MAJORITY])
                y_tie.append(st[ANSWER_VERIFIER_TIE])
                y_cheap.append(st[ANSWER_CHEAP_TIE])
                tied.append(st["tied"])
                lazy.append(st["n_scored_if_lazy"])
            slices.append((pi, start, len(vote)))
    return {"vote": np.array(vote), "score": np.array(score),
            "y": {ANSWER_MAJORITY: np.array(y_maj),
                  ANSWER_CHEAP_TIE: np.array(y_cheap),
                  ANSWER_VERIFIER_TIE: np.array(y_tie)},
            "tied": np.array(tied), "lazy": np.array(lazy, dtype=float),
            "slices": slices}


def design(flat: dict, which: str) -> np.ndarray:
    if which == "vote":
        return flat["vote"]
    if which == "score":
        return flat["score"]
    return np.hstack([flat["vote"], flat["score"]])


def stop_indices(probs: np.ndarray, taus: Sequence[float]) -> np.ndarray:
    """First draw clearing each tau, or the last draw when none does."""
    out = np.empty(len(taus), dtype=int)
    for j, tau in enumerate(taus):
        hit = np.flatnonzero(probs >= tau)
        out[j] = hit[0] if hit.size else len(probs) - 1
    return out


def run_policies(runs: dict[str, list], problems: list[str], question: dict[str, str],
                 n_folds: int, seed: int, budgets: list[float]) -> dict:
    """Every (stopping, answer) pair, out of fold, with thresholds picked in fold."""
    flat = flatten(runs, problems)
    slices = flat["slices"]
    n_orders = len(runs[problems[0]])
    n_train_orders = max(1, n_orders // 5)
    # Orders are split too: the stopping model is fitted on some sampling orders
    # and every threshold is chosen on the others, both from training problems.
    order_of = []
    seen: dict[int, int] = defaultdict(int)
    for pi, _, _ in slices:
        order_of.append(seen[pi])
        seen[pi] += 1
    order_of = np.array(order_of)
    fold_of = np.empty(len(problems), dtype=int)
    folds = grouped_folds([question[p] for p in problems], n_folds, seed)
    for k, f in enumerate(folds):
        fold_of[f] = k

    result = {a: {m: defaultdict(list) for m in [*STOP_MODELS, "fixed", "margin"]}
              for a in ANSWER_RULES}
    per_problem = {a: defaultdict(lambda: defaultdict(list)) for a in ANSWER_RULES}

    # Policies that read no fitted model are the same in every fold.
    for run_i, (pi, s0, s1) in enumerate(slices):
        p = problems[pi]
        T = s1 - s0
        margins = flat["vote"][s0:s1, 2] * np.round(flat["vote"][s0:s1, 0] * 10)
        for a in ANSWER_RULES:
            y = flat["y"][a][s0:s1]
            for n in sorted({*(int(b) for b in budgets), *range(1, 11)}):
                t = min(n, T) - 1
                result[a]["fixed"][("n", n)].append((t + 1, y[t]))
                if float(n) in budgets:
                    per_problem[a][("fixed", float(n))][p].append(y[t])
            for d in (1, 2, 3, 4):
                hit = np.flatnonzero(margins >= d)
                t = int(hit[0]) if hit.size else T - 1
                result[a]["margin"][("d", d)].append((t + 1, y[t]))

    for k in range(n_folds):
        train_rows = np.zeros(len(flat["vote"]), dtype=bool)
        fit_rows = np.zeros_like(train_rows)
        for pi, s0, s1 in slices:
            if fold_of[pi] != k:
                train_rows[s0:s1] = True
        for run_i, (pi, s0, s1) in enumerate(slices):
            if fold_of[pi] != k and order_of[run_i] < n_train_orders:
                fit_rows[s0:s1] = True
        tune_rows = train_rows & ~fit_rows
        for which in STOP_MODELS:
            X = design(flat, which)
            sc = StandardScaler().fit(X[fit_rows])
            model = LogisticRegression(max_iter=1000).fit(
                sc.transform(X[fit_rows]), flat["y"][ANSWER_VERIFIER_TIE][fit_rows])
            probs = model.predict_proba(sc.transform(X))[:, 1]
            # cost of each threshold, measured on held-out orders of training problems
            costs = np.zeros(len(TAUS))
            n_tune = 0
            stops = {}
            for run_i, (pi, s0, s1) in enumerate(slices):
                idx = stop_indices(probs[s0:s1], TAUS)
                stops[run_i] = idx
                if tune_rows[s0]:
                    costs += idx + 1
                    n_tune += 1
            mean_cost = costs / max(1, n_tune)
            chosen = {b: max((j for j, t in enumerate(TAUS) if mean_cost[j] <= b),
                             default=None) for b in budgets}
            for run_i, (pi, s0, s1) in enumerate(slices):
                if fold_of[pi] != k:
                    continue
                idx = stops[run_i]
                p = problems[pi]
                for a in ANSWER_RULES:
                    y = flat["y"][a][s0:s1]
                    for j, tau in enumerate(TAUS):
                        result[a][which][("tau", tau)].append((idx[j] + 1, y[idx[j]]))
                    for b in budgets:
                        j = chosen[b]
                        if j is None:
                            continue
                        result[a][which][("budget", b)].append((idx[j] + 1, y[idx[j]]))
                        per_problem[a][(which, b)][p].append(y[idx[j]])

    # Where the answer-rule gain comes from: the vote ties, and how often.
    by_n: dict[int, dict] = {}
    for n in range(1, 11):
        rows = np.array([s0 + min(n, s1 - s0) - 1 for _, s0, s1 in slices])
        by_n[n] = {
            "tie_rate": float(flat["tied"][rows].mean()),
            "mean_candidates_scored_if_lazy": float(flat["lazy"][rows].mean()),
            "majority": float(flat["y"][ANSWER_MAJORITY][rows].mean()),
            "majority_tie": float(flat["y"][ANSWER_VERIFIER_TIE][rows].mean()),
            "majority_cheap": float(flat["y"][ANSWER_CHEAP_TIE][rows].mean()),
            # A single draw never ties, so these are undefined at N=1.
            "tie_accuracy_chance": mean_or_none(
                flat["y"][ANSWER_MAJORITY][rows][flat["tied"][rows]]),
            "tie_accuracy_verifier": mean_or_none(
                flat["y"][ANSWER_VERIFIER_TIE][rows][flat["tied"][rows]]),
        }
        pids = [problems[pi] for pi, _, _ in slices]
        for label, rule in (("gain_vs_majority", ANSWER_VERIFIER_TIE),
                            ("cheap_gain_vs_majority", ANSWER_CHEAP_TIE),
                            ("gain_vs_cheap", None)):
            if rule is None:
                gain = (flat["y"][ANSWER_VERIFIER_TIE][rows]
                        - flat["y"][ANSWER_CHEAP_TIE][rows])
            else:
                gain = flat["y"][rule][rows] - flat["y"][ANSWER_MAJORITY][rows]
            agg: dict[str, list] = defaultdict(list)
            for p, g in zip(pids, gain):
                agg[p].append(g)
            keys = sorted(agg)
            by_n[n][label] = cluster_bootstrap(
                np.array([np.mean(agg[p]) for p in keys]),
                [question[p] for p in keys])

    summary = {}
    for a in ANSWER_RULES:
        summary[a] = {}
        for m, points in result[a].items():
            summary[a][m] = {f"{k[0]}={k[1]}":
                             {"cost": float(np.mean([c for c, _ in v])),
                              "accuracy": float(np.mean([y for _, y in v]))}
                             for k, v in sorted(points.items(), key=lambda kv: str(kv[0]))}
    per_problem = {a: {k: dict(v) for k, v in d.items()} for a, d in per_problem.items()}
    return {"summary": summary, "per_problem": per_problem, "by_n": by_n,
            "n_folds": n_folds}


def budget_table(summary: dict, per_problem: dict, problems: list[str],
                 question: dict[str, str], budgets: list[float]) -> dict:
    """Accuracy at each budget, and the paired gap against fixed-budget voting."""
    table = {}
    for a in ANSWER_RULES:
        rows = {}
        env = {m: frontier_envelope([(v["cost"], v["accuracy"])
                                     for v in summary[a][m].values()])
               for m in summary[a]}
        # The vote-only competitor is everything a vote can do: any fixed budget,
        # any margin rule, and the fitted vote-feature rule.
        vote_only = frontier_envelope(
            [pt for m in ("fixed", "margin", "vote") for pt in env[m]])
        for b in budgets:
            row = {"vote_only": accuracy_at_budget(vote_only, b)}
            for m in STOP_MODELS:
                row[m] = accuracy_at_budget(env[m], b)
            base = per_problem[a].get(("fixed", b))
            for m in STOP_MODELS:
                cur = per_problem[a].get((m, b))
                if not cur or not base:
                    continue
                shared = sorted(set(cur) & set(base))
                paired = np.array([np.mean(cur[p]) - np.mean(base[p]) for p in shared])
                row[f"{m}_vs_fixed"] = cluster_bootstrap(
                    paired, [question[p] for p in shared])
            rows[b] = row
        table[a] = {"budgets": rows,
                    "envelopes": {m: env[m] for m in env} | {"vote_only": vote_only}}
    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onpolicy_root", type=Path,
                   default=ROOT / "cot-checker-results/onpolicy_v1")
    p.add_argument("--reprobe_root", type=Path,
                   default=ROOT / "cot-checker-results/reprobe_v1")
    p.add_argument("--out_dir", type=Path, default=ROOT / "results/onpolicy_budget")
    p.add_argument("--arms", nargs="+", default=["frozen"],
                   choices=["frozen", "retrained"])
    p.add_argument("--styles", nargs="+", default=["verifier"],
                   choices=["verifier", "generation"])
    p.add_argument("--cell", default=None,
                   help="Score one named cell directory instead of the arm's "
                        "whole ensemble.")
    p.add_argument("--subset", default="clean", choices=["clean", "all"])
    p.add_argument("--n_orders", type=int, default=100,
                   help="Random sampling orders per problem. A fifth of them "
                        "trains the stopping model, the rest are scored.")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    pool, question = load_pool(args.onpolicy_root)
    outcomes = pool["outcomes"]
    uids = sorted(outcomes)
    cells = collect_cells(args.onpolicy_root, args.reprobe_root, args.arms, args.styles)
    if args.cell:
        cells = [c for c in cells if c[2].parent.name == args.cell]
    if not cells:
        raise SystemExit("no score files matched")
    L = load_scores([c[2] for c in cells], uids)
    quality = -L.mean(axis=0)

    answers = [normalize_answer(outcomes[u].get("pred")) for u in uids]
    correct = [bool(outcomes[u]["correct"]) for u in uids]
    members = defaultdict(list)
    for i, u in enumerate(uids):
        members[outcomes[u]["problem_id"]].append(i)

    fitted = fitted_questions(
        args.reprobe_root / "reprobe_train_judge_traces.jsonl",
        sorted(args.reprobe_root.glob("labels.shard*.jsonl")), 0, 0.15)
    problems = sorted(members)
    if args.subset == "clean":
        problems = [p for p in problems if question[p] not in fitted]
    print(f"[pool] {len(cells)} cells, {len(problems)} problems "
          f"({len(set(question[p] for p in problems))} question texts), "
          f"{args.n_orders} sampling orders each", flush=True)

    diag = margin_diagnostics(answers, correct, quality, members, problems)
    print(f"[margin] the vote is strong where the verifier is not needed: "
          f"{json.dumps(diag['by_margin'], default=float)}", flush=True)

    rng = np.random.default_rng(args.seed)
    # The free control: prefer the shorter solution. Length was the only cheap
    # ordering that came anywhere near the verifier at N=10 (§20.12).
    traj = pool["traj"]
    cheap = np.array([-len(traj[u]["solution"]) for u in uids], dtype=float)
    runs = {p: simulate(answers, correct, quality, members[p], args.n_orders, rng,
                        cheap=cheap)
            for p in problems}
    res = run_policies(runs, problems, question, args.n_folds, args.seed, BUDGETS)
    table = budget_table(res["summary"], res["per_problem"], problems, question,
                         BUDGETS)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "offline replay of saved candidates; no generation, no fitting "
                 "of any representation",
        "cells": [str(c[2].relative_to(ROOT)) for c in cells],
        "arms": args.arms, "styles": args.styles, "subset": args.subset,
        "n_problems": len(problems),
        "n_questions": len(set(question[p] for p in problems)),
        "n_orders": args.n_orders, "n_folds": args.n_folds, "seed": args.seed,
        "curves": res["summary"],
        "by_n": res["by_n"],
        "by_margin": diag["by_margin"],
        "budgets": {a: table[a]["budgets"] for a in ANSWER_RULES},
        "envelopes": {a: table[a]["envelopes"] for a in ANSWER_RULES},
        "source_sha256": {str(c[2].relative_to(ROOT)):
                          hashlib.sha256(c[2].read_bytes()).hexdigest()
                          for c in cells},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{'_'.join(args.arms)}_{'_'.join(args.styles)}_{args.subset}"
    (args.out_dir / f"budget_{tag}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False, default=float))
    text = render(report)
    (args.out_dir / f"budget_{tag}.md").write_text(text)
    print(text)
    print(f"[wrote] {args.out_dir}/budget_{tag}.json")
    print(f"[wrote] {args.out_dir}/budget_{tag}.md")


def render(report: dict) -> str:
    out = [f"\n## Accuracy under a sample budget "
           f"({len(report['cells'])} cell(s), {report['subset']} subset, "
           f"n={report['n_problems']} problems over {report['n_questions']} "
           f"question texts, {report['n_orders']} sampling orders)\n"]
    for a in ANSWER_RULES:
        out.append(f"\n### answer rule: {a}\n")
        out.append("| budget | vote-only best | stop on votes | stop on scores | "
                   "stop on both | both vs fixed-N | 95% CI |")
        out.append("|---|---|---|---|---|---|---|")
        for b, row in sorted(report["budgets"][a].items(), key=lambda kv: float(kv[0])):
            d = row.get("both_vs_fixed")
            gap = f"{d['delta']:+.4f}" if d else ""
            ci = f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else ""
            out.append(f"| {float(b):.0f} | {row['vote_only']:.4f} | {row['vote']:.4f} "
                       f"| {row['score']:.4f} | {row['both']:.4f} | {gap} | {ci} |")
    out.append("\n### fixed budget: what the verifier is worth at each N\n")
    out.append("| N | tie rate | majority | + shortest | + verifier | verifier "
               "gain | 95% CI | over shortest | equivalent N | scored (lazy) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    plain = {int(n): v["majority"] for n, v in report["by_n"].items()}
    for n, v in sorted(report["by_n"].items(), key=lambda kv: int(kv[0])):
        g, c = v["gain_vs_majority"], v["gain_vs_cheap"]
        out.append(f"| {int(n)} | {v['tie_rate']:.3f} | {v['majority']:.4f} | "
                   f"{v['majority_cheap']:.4f} | {v['majority_tie']:.4f} | "
                   f"{g['delta']:+.4f} | [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] | "
                   f"{c['delta']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                   f"{equivalent_n(plain, v['majority_tie'])} | "
                   f"{v['mean_candidates_scored_if_lazy']:.2f} |")
    out.append("\n### where each signal is informative (full pool, within-problem "
               "ranking)\n")
    out.append("| vote margin | problems | majority | oracle | verifier AUROC "
               "| agreement AUROC |")
    out.append("|---|---|---|---|---|---|")
    for k, v in report["by_margin"].items():
        out.append(f"| {k} | {v['n']} | {v['majority']:.3f} | {v['oracle']:.3f} | "
                   f"{v['within_auc_verifier']:.3f} | {v['within_auc_agreement']:.3f} |")
    out.append("\nEquivalent N is where plain self-consistency reaches the same "
               "accuracy, linearly interpolated; > 10 means it does not reach it "
               "inside this pool. The lazy column is how many candidates the "
               "verifier has to score if it runs only when the vote ties.")
    return "\n".join(out) + "\n"


def equivalent_n(plain: dict[int, float], target: float) -> str:
    """Where plain self-consistency reaches `target`, by linear interpolation."""
    ns = sorted(plain)
    for a, b in zip(ns, ns[1:]):
        if plain[a] <= target <= plain[b] and plain[b] > plain[a]:
            return f"{a + (target - plain[a]) / (plain[b] - plain[a]):.1f}"
    return f"> {ns[-1]}" if target > plain[ns[-1]] else f"< {ns[0]}"


if __name__ == "__main__":
    main()
