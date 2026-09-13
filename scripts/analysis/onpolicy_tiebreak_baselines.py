#!/usr/bin/env python3
"""Race the verifier's tie-break against free ones, on the saved on-policy scores.

The audit reduced the on-policy arm's one positive result to a single claim: on a
tied vote, the hidden-state score picks the winning answer slightly more often
than picking at random (REPORT.md §20.2, §20.11). That claim is worth something
only if the score also beats tie-breakers that cost nothing, so this recomputes
the decision three ways:

  1. split every problem into a unique plurality, where no selector can change
     the answer, and a co-plurality tie, where the whole effect lives;
  2. on the ties, compare the verifier against shortest, longest, fewest steps,
     most steps, first sampled, and the expectation of picking at random;
  3. repeat on the subset of evaluation questions whose text appears in neither
     the fitted training pool nor the validation set used to select on, with the
     bootstrap resampling question texts rather than problem ids.

CPU only, no fitting and no generation: everything below is a re-reading of files
already on disk.
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

from scripts.onpolicy.build_onpolicy_splits import (  # noqa: E402
    canonical_question, join_labels, split_by_problem,
)
from src.analysis.onpolicy_tiebreak import (  # noqa: E402
    RANDOM, SELECTORS, cluster_bootstrap, problem_record,
)
from src.eval.math_grade import normalize_answer  # noqa: E402

CHEAP = ["shortest", "longest", "fewest_steps", "most_steps", "first_sampled"]
VERIFIER = ["verifier_worst_step", "verifier_mean_step", "verifier_last_step"]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_pool(onpolicy_root: Path) -> tuple[dict, dict]:
    """The evaluation pool as it was actually scored, plus each problem's question."""
    outcomes = {r["id"]: r for r in read_rows(onpolicy_root / "onpolicy_stage1_outcomes.jsonl")}
    traj = {r["traj_uid"]: r
            for p in sorted(onpolicy_root.glob("*_trajectories.jsonl"))
            for r in read_rows(p)}
    question = {}
    for uid, o in outcomes.items():
        if uid in traj:
            question[o["problem_id"]] = canonical_question(traj[uid]["problem"])
    return {"outcomes": outcomes, "traj": traj}, question


def fitted_questions(judge_traces: Path, labels: list[Path], seed: int,
                     val_frac: float) -> set[str]:
    """Question texts the retrained heads were fitted on or selected on.

    Reconstructed from the archived split's own code and seed, because that is
    what the published cells actually saw. Rebuilding it a better way would answer
    a different question.
    """
    lab = {}
    for f in labels:
        for r in read_rows(f):
            lab[r.get("traj_uid") or r.get("id")] = r
    joined, _ = join_labels(read_rows(judge_traces), lab, min_steps=2)
    train, val, _ = split_by_problem(joined, val_frac, seed)
    return {canonical_question(t["problem"]) for t in (*train, *val)}


def candidates(score_path: Path, pool: dict) -> dict[str, list[dict]]:
    """Join one cell's step scores onto the recorded outcomes, by problem."""
    outcomes, traj = pool["outcomes"], pool["traj"]
    by_problem: dict[str, list[dict]] = defaultdict(list)
    joined = skipped = 0
    for row in read_rows(score_path):
        o = outcomes.get(row["id"])
        if o is None:
            skipped += 1
            continue
        if len(row["scores"]) != o["n_steps"]:
            raise SystemExit(f"{score_path}: {row['id']} has {len(row['scores'])} "
                             f"scores against {o['n_steps']} steps")
        joined += 1
        by_problem[o["problem_id"]].append({
            "answer": normalize_answer(o.get("pred")),
            "correct": bool(o["correct"]),
            "scores": [float(s) for s in row["scores"]],
            "n_chars": len(traj[row["id"]]["solution"]),
            "n_steps": int(o["n_steps"]),
            "index": row["id"],
        })
    for rows in by_problem.values():
        rows.sort(key=lambda r: r["index"])
        for i, r in enumerate(rows):
            r["index"] = i
    return {"by_problem": dict(by_problem), "joined": joined, "unknown_ids": skipped}


def summarise(records: dict[str, dict], question: dict[str, str],
              rules: list[str]) -> dict:
    """Accuracy of each rule on the ties, paired against picking at random."""
    tied = {pid: r for pid, r in records.items() if r["tied"]}
    out = {"n_problems": len(records), "n_tied": len(tied),
           "n_tied_questions": len({question[p] for p in tied}),
           "mixed_label_answer_groups": sorted(
               {f"{p}:{a}" for p, r in records.items()
                for a in r["mixed_label_answer_groups"]}),
           "oracle": float(np.mean([r["oracle"] for r in records.values()])),
           "rules": {}}
    if not tied:
        return out
    pids = sorted(tied)
    clusters = [question[p] for p in pids]
    base = np.array([tied[p]["accuracy"][RANDOM] for p in pids])
    out["tied_problem_ids"] = pids
    out["tie_accuracy_by_problem"] = {}
    for rule in [*rules, RANDOM]:
        acc = np.array([tied[p]["accuracy"][rule] for p in pids])
        out["tie_accuracy_by_problem"][rule] = acc.tolist()
        # Accuracy over every problem, using this rule only where the vote ties:
        # the number a downstream user would actually see.
        overall = float(np.mean([r["accuracy"][rule] for r in records.values()]))
        entry = {"tie_accuracy": float(acc.mean()), "overall_accuracy": overall}
        if rule != RANDOM:
            entry["vs_random"] = cluster_bootstrap(acc - base, clusters)
        out["rules"][rule] = entry
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onpolicy_root", type=Path,
                   default=ROOT / "cot-checker-results/onpolicy_v1")
    p.add_argument("--reprobe_root", type=Path,
                   default=ROOT / "cot-checker-results/reprobe_v1")
    p.add_argument("--out_dir", type=Path, default=ROOT / "results/onpolicy_tiebreak")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--split_val_frac", type=float, default=0.15)
    p.add_argument("--limit_cells", type=int, default=None,
                   help="Score files to read, for a smoke run.")
    args = p.parse_args()

    pool, question = load_pool(args.onpolicy_root)
    fitted = fitted_questions(
        args.reprobe_root / "reprobe_train_judge_traces.jsonl",
        sorted(args.reprobe_root.glob("labels.shard*.jsonl")),
        args.split_seed, args.split_val_frac)
    clean = {pid for pid, q in question.items() if q not in fitted}
    print(f"[pool] {len(question)} problems, {len(set(question.values()))} question "
          f"texts, {len(clean)} problems whose question was neither fitted nor "
          f"selected on", flush=True)

    score_files = []
    for arm, root in (("frozen", args.onpolicy_root / "grid"),
                      ("retrained", args.reprobe_root / "grid")):
        for style in ("verifier", "generation"):
            score_files += [(arm, style, f) for f in
                            sorted(root.rglob(f"*scores*{style}*.jsonl"))]
    if args.limit_cells:
        score_files = score_files[:args.limit_cells]
    if not score_files:
        raise SystemExit("no score files found")

    cells, cheap_done = [], False
    cheap_report: dict[str, dict] = {}
    agg: dict[str, dict] = {}
    digests = {}
    for arm, style, path in score_files:
        joined = candidates(path, pool)
        recs = {pid: problem_record(rows, VERIFIER + CHEAP)
                for pid, rows in joined["by_problem"].items()}
        row = {"arm": arm, "style": style, "cell": path.parent.name,
               "path": str(path.relative_to(ROOT)), "joined_rows": joined["joined"],
               "unknown_ids": joined["unknown_ids"]}
        for subset, keep in (("all", set(recs)), ("clean", clean & set(recs))):
            row[subset] = summarise({p: recs[p] for p in keep}, question, VERIFIER)
        for subset in ("all", "clean"):
            sub = row[subset]
            if not sub["rules"]:
                continue
            key = tuple(sub["tied_problem_ids"])
            if subset not in agg:
                agg[subset] = {"pids": key,
                               "sums": {r: np.zeros(len(key)) for r in VERIFIER},
                               "n": 0}
            elif agg[subset]["pids"] != key:
                raise SystemExit(f"{path}: a different set of problems ties; the "
                                 f"cells cannot be averaged per problem")
            for rule in VERIFIER:
                agg[subset]["sums"][rule] += np.array(
                    sub["tie_accuracy_by_problem"][rule])
            agg[subset]["n"] += 1
            # Only the aggregate needs the per-problem vector; keeping 228 copies
            # of it would triple the report for nothing.
            del sub["tie_accuracy_by_problem"]
        cells.append(row)
        digests[row["path"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not cheap_done:
            # The free rules read no scores, so they are identical in every cell.
            # Computing them once and checking that the pool never changes is
            # cheaper than recomputing 228 copies of the same number.
            for subset, keep in (("all", set(recs)), ("clean", clean & set(recs))):
                cheap_report[subset] = summarise({p: recs[p] for p in keep},
                                                 question, CHEAP)
            cheap_pool = sorted(recs)
            cheap_done = True
        elif sorted(recs) != cheap_pool:
            raise SystemExit(f"{path} scores a different problem set; the free "
                             f"baselines are no longer comparable across cells")
        print(f"[cell] {arm}/{style}/{path.parent.name} "
              f"tie n={row['all']['n_tied']} "
              f"verifier={row['all']['rules']['verifier_worst_step']['tie_accuracy']:.3f} "
              f"random={row['all']['rules'][RANDOM]['tie_accuracy']:.3f}", flush=True)

    # The average cell, rather than the best of 228. Each problem's accuracy is
    # averaged over cells first, so the interval is over problems (clustered by
    # question) with the cell held as a fixed effect.
    cell_averaged = {}
    for subset, a in agg.items():
        free = cheap_report[subset]
        base = np.array(free["tie_accuracy_by_problem"][RANDOM])
        clusters = [question[p] for p in a["pids"]]
        cell_averaged[subset] = {
            rule: {"tie_accuracy": float((a["sums"][rule] / a["n"]).mean()),
                   "vs_random": cluster_bootstrap(a["sums"][rule] / a["n"] - base,
                                                  clusters),
                   "n_cells": a["n"]}
            for rule in VERIFIER}

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "recomputation of saved scores; no fitting, no generation",
        "n_cells": len(cells),
        "n_problems": len(question),
        "n_question_texts": len(set(question.values())),
        "n_clean_problems": len(clean),
        "fitted_or_selected_question_texts": len(fitted),
        "free_baselines": cheap_report,
        "cell_averaged_verifier": cell_averaged,
        "cells": cells,
        "source_sha256": digests,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tiebreak_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False))
    (args.out_dir / "tiebreak_table.md").write_text(render(report))
    print(render(report))
    print(f"[wrote] {args.out_dir}/tiebreak_results.json")
    print(f"[wrote] {args.out_dir}/tiebreak_table.md")


def render(report: dict) -> str:
    lines = []
    for subset in ("all", "clean"):
        free = report["free_baselines"][subset]
        lines.append(f"\n### Tie-break, {subset} problems "
                     f"({free['n_tied']} ties over {free['n_tied_questions']} "
                     f"question texts, out of {free['n_problems']})\n")
        lines.append("| rule | tie accuracy | overall | delta vs random | 95% CI |")
        lines.append("|---|---|---|---|---|")
        r = free["rules"][RANDOM]
        lines.append(f"| random (expectation) | {r['tie_accuracy']:.4f} | "
                     f"{r['overall_accuracy']:.4f} | | |")
        for rule in CHEAP:
            e = free["rules"][rule]
            b = e["vs_random"]
            lines.append(f"| {rule} | {e['tie_accuracy']:.4f} | "
                         f"{e['overall_accuracy']:.4f} | {b['delta']:+.4f} | "
                         f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] |")
        for rule in VERIFIER:
            vals = [(c[subset]["rules"][rule]["tie_accuracy"], c) for c in report["cells"]]
            best_acc, best = max(vals, key=lambda t: t[0])
            acc = np.array([v for v, _ in vals])
            ca = report["cell_averaged_verifier"][subset][rule]
            b_ca = ca["vs_random"]
            lines.append(f"| {rule} (averaged over {len(vals)} cells) | "
                         f"{ca['tie_accuracy']:.4f} | | {b_ca['delta']:+.4f} | "
                         f"[{b_ca['ci95'][0]:+.4f}, {b_ca['ci95'][1]:+.4f}] |")
            lines.append(f"| {rule} (median / min / max over cells) | "
                         f"{np.median(acc):.4f} / {acc.min():.4f} / {acc.max():.4f} "
                         f"| | | |")
            b = best[subset]["rules"][rule]["vs_random"]
            lines.append(f"| {rule} (best cell*: {best['arm']}/{best['style']}/"
                         f"{best['cell']}) | {best_acc:.4f} | "
                         f"{best[subset]['rules'][rule]['overall_accuracy']:.4f} | "
                         f"{b['delta']:+.4f} | [{b['ci95'][0]:+.4f}, "
                         f"{b['ci95'][1]:+.4f}] |")
        lines.append("")
        lines.append("\\* The best cell is the maximum over "
                     f"{len(report['cells'])} cells of a quantity measured on the "
                     "same problems, so its interval is not a test of anything; "
                     "the mean row is the one to compare against the free rules.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
