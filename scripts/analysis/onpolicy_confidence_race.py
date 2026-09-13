#!/usr/bin/env python3
"""Race the verifier's tie-break against token confidence, the missing competitor.

REPORT.md §20.12 established that the hidden-state verifier is the only
tie-break rule whose interval clears zero, and closed with the limitation that
made the result provisional: "the obvious remaining competitor, token
confidence, cannot be computed at all from what is on disk, because the saved
trajectories carry no logprobs. Until that comparison exists, 'beats free' means
'beats free *length* rules'."

scripts/onpolicy/encode_token_confidence.py recovers those logprobs from the
saved trajectory text, so this script closes the gap. It is the Phase 1 gate of
docs/onpolicy_tiebreak_v2_plan.md: if a free DeepConf-style statistic breaks
ties as well as the verifier, the hidden states buy nothing over logprobs and
the study reports that.

Two rules of the comparison, both of which make it harder for us to win:

  * the confidence baseline gets both orientations of every statistic and is
    scored at its best, because a competitor crippled by a sign is not a
    competitor;
  * one confidence rule, `step_min_conf`, uses the verifier's own
    minimum-over-steps aggregation, so at least one contrast varies the signal
    with the aggregation held fixed.

CPU only. Everything read here is already on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.onpolicy_tiebreak_baselines import (  # noqa: E402
    candidates, fitted_questions, load_pool, read_rows, summarise,
)
from src.analysis.onpolicy_tiebreak import (  # noqa: E402
    RANDOM, best_confidence_rule, problem_record, register_confidence_selectors,
)

VERIFIER = ["verifier_worst_step", "verifier_mean_step", "verifier_last_step"]
CHEAP = ["shortest", "longest", "fewest_steps", "most_steps", "first_sampled"]


def load_confidence(paths: list[Path]) -> tuple[dict[str, dict], list[str]]:
    """Per-trajectory confidence scalars, keyed by traj_uid."""
    by_uid: dict[str, dict] = {}
    names: set[str] = set()
    for p in paths:
        for row in read_rows(p):
            vals = dict(row["rules"])
            by_uid[row["traj_uid"]] = vals
            names.update(vals)
    return by_uid, sorted(names)


def attach_confidence(by_problem: dict[str, list[dict]],
                      conf: dict[str, dict]) -> dict:
    """Join confidence onto candidates, and report what did not join.

    `candidates` renumbers `index` to a within-problem position, so the join key
    has to be captured before that happens; this reads `uid`, which
    `candidates_with_uid` below preserves for exactly this reason.
    """
    hit = miss = 0
    for rows in by_problem.values():
        for r in rows:
            v = conf.get(r["uid"])
            r["conf"] = v or {}
            hit, miss = (hit + 1, miss) if v else (hit, miss + 1)
    return {"joined": hit, "missing": miss}


def candidates_with_uid(score_path: Path, pool: dict) -> dict:
    """`candidates`, keeping the trajectory uid the confidence file is keyed by."""
    out = candidates(score_path, pool)
    outcomes = pool["outcomes"]
    uid_by_problem: dict[str, list[str]] = defaultdict(list)
    for row in read_rows(score_path):
        o = outcomes.get(row["id"])
        if o is not None:
            uid_by_problem[o["problem_id"]].append(row["id"])
    for pid, rows in out["by_problem"].items():
        uids = sorted(uid_by_problem[pid])
        if len(uids) != len(rows):
            raise SystemExit(f"{score_path}: {pid} has {len(rows)} candidates "
                             f"against {len(uids)} uids")
        for r, uid in zip(rows, uids):
            r["uid"] = uid
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onpolicy_root", type=Path,
                   default=ROOT / "cot-checker-results/onpolicy_v1")
    p.add_argument("--reprobe_root", type=Path,
                   default=ROOT / "cot-checker-results/reprobe_v1")
    p.add_argument("--confidence", type=Path, nargs="+", required=True,
                   help="*_conf.jsonl written by encode_token_confidence.py")
    p.add_argument("--out_dir", type=Path,
                   default=ROOT / "results/onpolicy_confidence_race")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--split_val_frac", type=float, default=0.15)
    p.add_argument("--limit_cells", type=int, default=None)
    args = p.parse_args()

    pool, question = load_pool(args.onpolicy_root)
    fitted = fitted_questions(
        args.reprobe_root / "reprobe_train_judge_traces.jsonl",
        sorted(args.reprobe_root.glob("labels.shard*.jsonl")),
        args.split_seed, args.split_val_frac)
    clean = {pid for pid, q in question.items() if q not in fitted}

    conf, conf_names = load_confidence(args.confidence)
    conf_rules = register_confidence_selectors(conf_names)
    print(f"[conf] {len(conf)} trajectories, {len(conf_names)} statistics, "
          f"{len(conf_rules)} selectors (both orientations)", flush=True)

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

    rules = VERIFIER + CHEAP + conf_rules
    per_cell, join_stats = [], []
    for arm, style, path in score_files:
        joined = candidates_with_uid(path, pool)
        join_stats.append(attach_confidence(joined["by_problem"], conf))
        recs = {pid: problem_record(rows, rules)
                for pid, rows in joined["by_problem"].items()
                if pid in clean}
        per_cell.append({"arm": arm, "style": style, "cell": path.stem,
                         "summary": summarise(recs, question, rules)})
        print(f"[cell] {len(per_cell)}/{len(score_files)} {path.stem}", flush=True)

    # Average each rule's tie accuracy over cells per problem, as §20.12 does, so
    # the reported number is the typical cell and not the best of many.
    pooled: dict[str, list[float]] = defaultdict(list)
    for c in per_cell:
        for rule, entry in c["summary"].get("rules", {}).items():
            pooled[rule].append(entry["tie_accuracy"])
    table = {r: {"tie_accuracy_mean_over_cells": float(np.mean(v)),
                 "n_cells": len(v)} for r, v in pooled.items()}

    winner = best_confidence_rule({"rules": {r: {"tie_accuracy": v[
        "tie_accuracy_mean_over_cells"]} for r, v in table.items()}}, conf_rules)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(per_cell), "n_clean_problems": len(clean),
        "confidence_statistics": conf_names,
        "join": {"joined": sum(j["joined"] for j in join_stats),
                 "missing": sum(j["missing"] for j in join_stats)},
        "table": table,
        "best_confidence_rule": winner,
        "verdict": {
            "verifier_worst_step": table.get("verifier_worst_step"),
            "best_confidence": table.get(winner) if winner else None,
            "random": table.get(RANDOM),
        },
        "cells": per_cell,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "confidence_race.json").write_text(json.dumps(report, indent=2))

    lines = ["| rule | tie accuracy (mean over cells) | cells |", "|---|---|---|"]
    for rule, v in sorted(table.items(), key=lambda kv: -kv[1][
            "tie_accuracy_mean_over_cells"]):
        lines.append(f"| {rule} | {v['tie_accuracy_mean_over_cells']:.4f} "
                     f"| {v['n_cells']} |")
    (args.out_dir / "confidence_race.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"[out] {args.out_dir/'confidence_race.json'}")


if __name__ == "__main__":
    main()
