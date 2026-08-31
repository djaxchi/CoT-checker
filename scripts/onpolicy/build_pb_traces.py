#!/usr/bin/env python3
"""Turn generated trajectories into a ProcessBench-format evaluation set.

The on-policy arm needs its steps to arrive at the harness the way ProcessBench's
do, because that is the path the whole grid is already scored on: a traces jsonl
goes into scripts/encode_processbench_token_store.py, which writes meta rows
carrying ``id`` / ``step_idx`` / ``label`` / ``n_steps``, which is exactly what
``evaluate_processbench`` groups on. Emitting that format here means the
on-policy split is just another ``--pb_subsets`` entry and no evaluation code
changes at all.

Two things have to be fixed on the way.

**The labels.** The generator inherits the trajectory's outcome onto every one of
its steps, which is right for the distribution control it was written for and
wrong here: F1_PB scores *first-error localisation*, and the early steps of a
wrong solution are usually fine. So this reads a judge's per-trajectory
first-error index (-1 for no error) and writes that as the trace label.

**The conflicts between the judge and the grader.** A trajectory that reached the
correct final answer should have no first error, and one that reached a wrong
answer should have one. Where the judge disagrees, the default policy trusts the
grader, because the grader is arithmetic against a known answer and the judge is
a model. Both conflict rates are counted into the manifest: the rate on correct
trajectories is the judge's false-alarm rate measured on exactly the distribution
the arm cares about, and it needs no human labels to compute.

Outputs:
  {stem}_pb_traces.jsonl   ProcessBench schema: id, problem, steps, label
  {stem}_outcomes.jsonl    one row per trajectory for the T2 simulations
                           (problem_id, correct, gradeable, pred, n_steps)
  {stem}_pb_manifest.json  counts, conflict rates, policies, code commit
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl, write_jsonl  # noqa: E402
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402

NO_ERROR = -1


def read_labels(paths: list[Path], min_base_rate: float = 0.0
                ) -> tuple[dict[str, int], Counter]:
    """traj_uid -> first-error step index, or -1, from a labeller's output.

    `min_base_rate` drops rollout labels for problems the model could not solve
    before the trajectory even started. A rollout labeller marks the first step
    after which no continuation reaches the answer, so on a problem it never
    solves, every step qualifies and the rule fires at step 0 for reasons that
    have nothing to do with the step. Those trajectories carry no information
    about where the reasoning went wrong and are dropped rather than labelled.
    Judge output carries no base_rate and is unaffected.
    """
    out: dict[str, int] = {}
    tally: Counter = Counter()
    for p in paths:
        for row in read_jsonl(p):
            uid = row.get("traj_uid") or row.get("id")
            if uid is None:
                raise ValueError(f"{p}: label row without traj_uid/id: {row}")
            base = row.get("base_rate")
            if base is not None and float(base) < min_base_rate:
                tally["dropped_unsolvable"] += 1
                continue
            if uid in out and out[uid] != int(row["first_error"]):
                raise ValueError(f"{p}: conflicting labels for {uid}")
            out[uid] = int(row["first_error"])
            tally["labels_read"] += 1
    return out, tally


def resolve_label(judged: int | None, correct: bool, n_steps: int,
                  correct_policy: str, no_error_policy: str,
                  unjudged_correct: str = "drop") -> tuple[int | None, str]:
    """(label, tag). label None means drop the trace. tag counts into the manifest."""
    if judged is None:
        # A trajectory that reached the right answer already has a label from the
        # grader, so it can join the evaluation set without a judge ever seeing
        # it. That is not free of assumption: a correct final answer can follow a
        # wrong step the model later repaired, and those traces would be scored
        # as clean. The audited sample measures how often that happens, and the
        # label source is recorded per trace so the analysis can be run with and
        # without them.
        if correct and unjudged_correct == "no_error":
            return NO_ERROR, "unjudged_correct_from_grader"
        return None, "unjudged"
    if judged != NO_ERROR and not 0 <= judged < n_steps:
        return None, "label_out_of_range"
    if correct:
        if judged == NO_ERROR:
            return NO_ERROR, "agree_correct"
        if correct_policy == "trust_outcome":
            return NO_ERROR, "conflict_correct_overridden"
        if correct_policy == "trust_judge":
            return judged, "conflict_correct_kept"
        return None, "conflict_correct_dropped"
    if judged != NO_ERROR:
        return judged, "agree_error"
    # wrong answer, judge found nothing
    if no_error_policy == "last_step":
        return n_steps - 1, "conflict_error_last_step"
    return None, "conflict_error_dropped"


def build(trajectories: list[dict], labels: dict[str, int], correct_policy: str,
          no_error_policy: str, min_steps: int, unjudged_correct: str = "drop"
          ) -> tuple[list[dict], list[dict], Counter]:
    traces, outcomes, tally = [], [], Counter()
    for tr in trajectories:
        uid = tr["traj_uid"]
        tally["trajectories"] += 1
        if not tr.get("gradeable", False):
            tally["ungradeable"] += 1
            continue
        steps = split_into_steps(tr["solution"])
        if len(steps) < min_steps:
            tally["too_few_steps"] += 1
            continue
        label, tag = resolve_label(labels.get(uid), bool(tr["correct"]), len(steps),
                                   correct_policy, no_error_policy, unjudged_correct)
        tally[tag] += 1
        outcomes.append({
            "id": uid, "problem_id": tr["fork_id"], "correct": bool(tr["correct"]),
            "gradeable": True, "pred": tr.get("pred"), "gold": tr.get("gold"),
            "n_steps": len(steps), "in_pb_traces": label is not None,
        })
        if label is None:
            continue
        traces.append({
            "id": uid, "problem": tr["problem"], "steps": steps, "label": int(label),
            "problem_id": tr["fork_id"], "traj_correct": bool(tr["correct"]),
            "n_steps": len(steps),
            "label_source": "grader" if tag.startswith("unjudged_correct") else "judge",
        })
        tally["kept"] += 1
    return traces, outcomes, tally


def judge_traces(trajectories: list[dict], min_steps: int, correct_sample: int,
                 seed: int = 0, max_incorrect: int = 0) -> list[dict]:
    """The traces to send to a judge, before any labels exist.

    Only incorrect trajectories need a verdict: a correct one takes -1 from the
    grader under the default policy, and asking a judge about it buys nothing but
    cost. A sample of correct ones goes anyway, because how often the judge
    invents an error in work that reached the right answer is its false-alarm
    rate on the distribution this arm cares about, and it needs no human labels.
    """
    import random
    rng = random.Random(seed)
    wrong, right = [], []
    for tr in trajectories:
        if not tr.get("gradeable"):
            continue
        steps = split_into_steps(tr["solution"])
        if len(steps) < min_steps:
            continue
        row = {"id": tr["traj_uid"], "problem": tr["problem"], "steps": steps,
               "traj_correct": bool(tr["correct"]), "gold": tr.get("gold"),
               "problem_id": tr["fork_id"], "n_steps": len(steps)}
        (right if tr["correct"] else wrong).append(row)
    rng.shuffle(right)
    rng.shuffle(wrong)
    if max_incorrect > 0:
        wrong = wrong[:max_incorrect]
    return wrong + right[:correct_sample]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", nargs="+", required=True, type=Path,
                   help="One or more *_trajectories.jsonl (the generator's shards).")
    p.add_argument("--labels", nargs="*", default=[], type=Path,
                   help="Judge output: rows of {traj_uid, first_error}. Without it "
                        "only the outcomes sidecar is written, which is all the T2 "
                        "simulations need.")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--stem", default="onpolicy")
    p.add_argument("--correct_traj_policy",
                   choices=["trust_outcome", "trust_judge", "drop_conflict"],
                   default="trust_outcome",
                   help="What to do when the judge finds an error in a trajectory "
                        "that reached the right answer.")
    p.add_argument("--incorrect_no_error_policy", choices=["drop", "last_step"],
                   default="drop",
                   help="What to do when the judge finds no error in a trajectory "
                        "that reached the wrong answer.")
    p.add_argument("--unjudged_correct", choices=["drop", "no_error"], default="drop",
                   help="What to do with correct trajectories no judge saw. "
                        "'no_error' takes the grader's verdict, which is how a "
                        "budgeted judge run still yields a full evaluation set; "
                        "every such trace is marked label_source=grader.")
    p.add_argument("--min_base_rate", type=float, default=0.0,
                   help="Drop rollout labels whose pre-trajectory solve rate is "
                        "below this. A problem the model never solves makes every "
                        "step look unrecoverable, so the label says nothing about "
                        "the reasoning. Set above 0 for rollout labels; judge "
                        "labels have no base rate and are unaffected.")
    p.add_argument("--min_steps", type=int, default=2,
                   help="A one-step solution carries no localisation signal.")
    p.add_argument("--for_judge", type=Path, default=None,
                   help="Also write the traces a judge should read, before any "
                        "labels exist: every incorrect trajectory plus a sample "
                        "of correct ones for the false-alarm measurement.")
    p.add_argument("--judge_correct_sample", type=int, default=200)
    p.add_argument("--judge_max_incorrect", type=int, default=0,
                   help="Cap the incorrect trajectories sent to the judge, drawn "
                        "at random under --seed. A paid judge has a budget; the "
                        "correct trajectories not sent still enter the eval set "
                        "with the grader's -1.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = args.out_dir / f"{args.stem}_pb_traces.jsonl"
    if traces_path.exists() and not args.force:
        sys.exit(f"[pb_traces] refusing to overwrite {traces_path}; pass --force")

    trajectories: list[dict] = []
    seen: set[str] = set()
    for f in args.trajectories:
        for tr in read_jsonl(f):
            if tr["traj_uid"] in seen:
                sys.exit(f"[pb_traces] duplicate traj_uid {tr['traj_uid']} in {f}; "
                         f"shards must partition the problem list")
            seen.add(tr["traj_uid"])
            trajectories.append(tr)

    if args.for_judge:
        jt = judge_traces(trajectories, args.min_steps, args.judge_correct_sample,
                          args.seed, args.judge_max_incorrect)
        write_jsonl(args.for_judge, jt)
        n_wrong = sum(1 for t in jt if not t["traj_correct"])
        print(f"[pb_traces] {len(jt)} traces for the judge "
              f"({n_wrong} incorrect + {len(jt)-n_wrong} correct audited) "
              f"-> {args.for_judge}")

    labels, label_tally = read_labels(list(args.labels), args.min_base_rate)
    traces, outcomes, tally = build(trajectories, labels, args.correct_traj_policy,
                                    args.incorrect_no_error_policy, args.min_steps,
                                    args.unjudged_correct)

    if labels:
        write_jsonl(traces_path, traces)
    write_jsonl(args.out_dir / f"{args.stem}_outcomes.jsonl", outcomes)

    n_correct = sum(o["correct"] for o in outcomes)
    fa_denom = tally["agree_correct"] + tally["conflict_correct_overridden"] + \
        tally["conflict_correct_kept"] + tally["conflict_correct_dropped"]
    manifest = {
        "stem": args.stem,
        "sources": [str(f) for f in args.trajectories],
        "label_files": [str(f) for f in args.labels],
        "policies": {"correct_traj": args.correct_traj_policy,
                     "incorrect_no_error": args.incorrect_no_error_policy,
                     "unjudged_correct": args.unjudged_correct,
                     "min_steps": args.min_steps},
        "label_sources": dict(Counter(t["label_source"] for t in traces)),
        "counts": dict(sorted(tally.items())),
        "label_file_counts": dict(sorted(label_tally.items())),
        "min_base_rate": args.min_base_rate,
        "n_traces": len(traces), "n_outcomes": len(outcomes),
        "n_problems": len({o["problem_id"] for o in outcomes}),
        "trajectory_accuracy": (n_correct / len(outcomes)) if outcomes else 0.0,
        # The judge's false-alarm rate on correct on-policy trajectories: an error
        # claimed where the arithmetic says the answer is right. Measured on the
        # distribution the arm cares about and free of human labels.
        "judge_false_alarm_rate": (
            (tally["conflict_correct_overridden"] + tally["conflict_correct_kept"]
             + tally["conflict_correct_dropped"]) / fa_denom) if fa_denom else None,
        "judge_miss_rate": (
            (tally["conflict_error_dropped"] + tally["conflict_error_last_step"]) /
            max(1, tally["agree_error"] + tally["conflict_error_dropped"]
                + tally["conflict_error_last_step"])) if labels else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }
    (args.out_dir / f"{args.stem}_pb_manifest.json").write_text(json.dumps(manifest, indent=2))
    for k, v in manifest["counts"].items():
        print(f"[pb_traces] {k:34s} {v}")
    print(f"[pb_traces] accuracy {manifest['trajectory_accuracy']:.3f}  "
          f"traces {len(traces)}  outcomes {len(outcomes)}")
    if labels:
        print(f"[pb_traces] judge false alarm {manifest['judge_false_alarm_rate']:.3f}  "
              f"miss {manifest['judge_miss_rate']:.3f}")
        print(f"[pb_traces] wrote {traces_path}")
    else:
        print("[pb_traces] no --labels given: outcomes sidecar only, no pb_traces")


if __name__ == "__main__":
    main()
