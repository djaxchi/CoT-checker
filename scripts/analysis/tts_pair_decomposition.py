"""Exact N=2 opportunity and ranking decomposition on saved candidate pools.

This is a conditional sensitivity analysis. Exclude whole questions with missing
answers, inconsistent labels within an answer group, or incomplete scores. Keep
an exclusion ledger; do not treat this subset as a repaired full-pool evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.tts_rule_contrast import paired_bootstrap  # noqa: E402
from src.eval.math_grade import normalize_answer  # noqa: E402


def decompose_pair_pool(answers: list[str], correct: list[bool],
                        quality: list[float]) -> dict[str, float]:
    """Average all unordered pairs, splitting exact score ties uniformly.

    Higher quality wins. Equal-label pairs cannot change correctness. With
    consistent, parseable answers, this equals averaging both draw orders under
    the frontier's first-in-order score tie policy.
    """
    if len(answers) < 2 or not (len(answers) == len(correct) == len(quality)):
        raise ValueError("Need at least two aligned candidates")
    if any(a is None for a in answers) or not np.isfinite(quality).all():
        raise ValueError("Answers and scores must be complete")
    labels: dict[str, bool] = {}
    for answer, label in zip(answers, correct):
        if answer in labels and labels[answer] != label:
            raise ValueError("One answer group has conflicting correctness labels")
        labels[answer] = label
    rows = []
    for i, j in itertools.combinations(range(len(answers)), 2):
        mixed = correct[i] != correct[j]
        majority = (float(correct[i]) + float(correct[j])) / 2
        selected = (majority if quality[i] == quality[j]
                    else float(correct[i] if quality[i] > quality[j] else correct[j]))
        rows.append([majority, selected, float(correct[i] or correct[j]),
                     float(mixed), float(mixed) * selected,
                     float(answers[i] != answers[j])])
    values = np.mean(rows, axis=0)
    return dict(zip(("majority", "selected", "oracle", "mixed_pair_rate",
                     "mixed_selected_mass", "answer_disagreement_rate"),
                    map(float, values)))


def read_rows(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text().splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cell", required=True, help="Score filename cell, without stem/shard")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"protocol": "exact unordered pairs, uniform score ties, equal question weight",
              "cell": args.cell, "datasets": {}, "sha256": {}}
    for stem in ("tts_gsm8k", "tts_math500"):
        trajectories = sorted(args.run_root.glob(f"{stem}.shard*_trajectories.jsonl"))
        score_paths = sorted((args.run_root / "scores").glob(f"{stem}__{args.cell}.shard*.jsonl"))
        if not trajectories or not score_paths:
            raise ValueError(f"Missing trajectories or scores for {stem}")
        for path in trajectories + score_paths:
            result["sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        score_rows = read_rows(score_paths)
        scores = {row["traj_uid"]: row["scores"] for row in score_rows}
        if len(scores) != len(score_rows):
            raise ValueError("Duplicate score trajectory IDs")
        by_problem = defaultdict(list)
        seen = set()
        for row in read_rows(trajectories):
            if row["traj_uid"] in seen:
                raise ValueError("Duplicate trajectory IDs")
            seen.add(row["traj_uid"])
            by_problem[row["fork_id"]].append(row)
        kept, excluded = {}, {}
        for pid, rows in sorted(by_problem.items()):
            answers = [normalize_answer(row.get("pred")) for row in rows]
            values = [scores.get(row["traj_uid"], []) for row in rows]
            if len(rows) != 10:
                excluded[pid] = "incomplete ten-candidate pool"
                continue
            if any(not row.get("gradeable") for row in rows):
                excluded[pid] = "ungradeable candidate"
                continue
            if any(not val or not np.isfinite(val).all() for val in values):
                excluded[pid] = "missing or nonfinite scores"
                continue
            try:
                stats = decompose_pair_pool(answers, [bool(r["correct"]) for r in rows],
                                            [-max(val) for val in values])
            except ValueError as error:
                excluded[pid] = str(error)
                continue
            kept[pid] = {**stats, "question_hash": rows[0].get("question_hash", pid)}
        if not kept:
            raise ValueError(f"No eligible questions for {stem}")
        mean = {key: float(np.mean([row[key] for row in kept.values()]))
                for key in stats}
        majority = np.array([row["majority"] for row in kept.values()])
        selected = np.array([row["selected"] for row in kept.values()])
        lift, lo, hi = paired_bootstrap(selected, majority,
                                        [row["question_hash"] for row in kept.values()])
        opportunity = mean["mixed_pair_rate"]
        mean.update(lift=lift, lift_ci95=[lo, hi],
                    conditional_mixed_accuracy=(mean["mixed_selected_mass"] / opportunity
                                                if opportunity else None),
                    headroom_recovered=(2 * lift / opportunity if opportunity else None))
        result["datasets"][stem] = {"n_total": len(by_problem), "n_kept": len(kept),
                                    "excluded": excluded, "summary": mean,
                                    "per_problem": kept}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "per_problem"}
                      for key, value in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()
