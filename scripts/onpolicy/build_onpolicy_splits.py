#!/usr/bin/env python3
"""Split the labelled on-policy pool into train, validation and test, by question.

Splitting by trajectory would put ten samples of the same problem on both sides
and let the probe memorise the problem rather than learn what a faulty step looks
like. The first version of this script therefore split over `problem_id`, which
turned out not to be enough: `build_prm800k_prestudy.py` mints fallback ids from
a sample index and a problem hash, so the same question text can carry two ids
and pass an id-level disjointness check. The 2026-09-10 audit measured the
damage on the archived split: 27 question texts shared between train and
validation, 49 between train and the frozen evaluation pool, and 16 between
validation and that pool, leaving only 228 of 300 evaluation questions unseen.

The unit is therefore the canonical question text (whitespace collapsed), and
every group of trajectories sharing that text lands on exactly one side.

Three splits are emitted, not two. The archived run trained with
`--test_stem val`, so its reported in-domain test number was its own model
selection set; an in-domain test that means anything needs a third split that
nothing selects on.

Emits ProcessBench-shaped traces carrying `step_labels`, which the encoder reads
to mark every faulty step rather than only a first error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl, write_jsonl  # noqa: E402


def join_labels(traces: list[dict], labels: dict[str, dict], min_steps: int
                ) -> tuple[list[dict], Counter]:
    """Attach each judged trajectory's step labels to its trace."""
    out, tally = [], Counter()
    for tr in traces:
        tally["traces"] += 1
        lab = labels.get(tr["id"])
        if lab is None:
            tally["unlabelled"] += 1
            continue
        if not lab.get("parse_ok"):
            tally["judge_parse_failed"] += 1
            continue
        sl = lab.get("step_labels")
        if sl is None or len(sl) != len(tr["steps"]):
            # A judge that named steps of a different length was not looking at
            # this trajectory; attaching the vector anyway would train on labels
            # belonging to something else.
            tally["label_length_mismatch"] += 1
            continue
        faulty = [i for i, v in enumerate(sl) if v == 0]
        out.append({**tr, "step_labels": sl,
                    "label": min(faulty) if faulty else -1,
                    "faulty_steps": faulty,
                    "label_source": "gpt-oss-reprobe"})
        tally["kept"] += 1
        tally["with_a_faulty_step"] += int(bool(faulty))
    return out, tally


def canonical_question(text: str) -> str:
    """The identity a split has to respect: the question itself, not its id.

    Whitespace-collapsed exact text. This under-counts duplication, since it
    misses paraphrases and re-typeset LaTeX, so every overlap it reports is real
    and the count it reports is a floor.
    """
    return " ".join(text.split())


def split_by_problem(traces: list[dict], val_frac: float, seed: int
                     ) -> tuple[list[dict], list[dict], dict]:
    """The historical id-level split. Retained only so the archived counts can be
    reproduced; `split_by_question` is the one to use. See the module docstring."""
    problems = sorted({t["problem_id"] for t in traces})
    rng = random.Random(seed)
    rng.shuffle(problems)
    n_val = max(1, int(round(len(problems) * val_frac)))
    val_ids = set(problems[:n_val])
    train = [t for t in traces if t["problem_id"] not in val_ids]
    val = [t for t in traces if t["problem_id"] in val_ids]
    return train, val, {"n_problems": len(problems), "n_val_problems": len(val_ids),
                        "val_problem_ids": sorted(val_ids), "seed": seed,
                        "val_frac": val_frac}


def split_by_question(traces: list[dict], val_frac: float, test_frac: float,
                      seed: int) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Assign whole canonical-question groups to train, validation and test.

    Deterministic under the seed. The returned info records the assignment by
    question digest rather than by text, so a manifest can prove which split a
    later run used without carrying every problem statement.
    """
    groups: dict[str, list[dict]] = {}
    for t in traces:
        groups.setdefault(canonical_question(t["problem"]), []).append(t)
    questions = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(questions)
    n_test = max(1, int(round(len(questions) * test_frac))) if test_frac > 0 else 0
    n_val = max(1, int(round(len(questions) * val_frac))) if val_frac > 0 else 0
    if n_test + n_val >= len(questions):
        raise SystemExit(f"val_frac + test_frac leave no training questions "
                         f"({len(questions)} questions available)")
    test_q = questions[:n_test]
    val_q = questions[n_test:n_test + n_val]
    train_q = questions[n_test + n_val:]
    out = tuple([t for q in qs for t in groups[q]] for qs in (train_q, val_q, test_q))
    info = {"unit": "canonical_question", "seed": seed,
            "val_frac": val_frac, "test_frac": test_frac,
            "n_questions": len(questions),
            "n_train_questions": len(train_q), "n_val_questions": len(val_q),
            "n_test_questions": len(test_q),
            "n_problem_ids": len({t["problem_id"] for t in traces}),
            "question_digests": {name: sorted(question_digest(q) for q in qs)
                                 for name, qs in (("train", train_q), ("val", val_q),
                                                  ("test", test_q))}}
    return (*out, info)


def question_digest(canonical_text: str) -> str:
    return hashlib.sha256(canonical_text.encode()).hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, type=Path,
                   help="the judge pool: id, problem, steps, gold, traj_correct")
    p.add_argument("--labels", nargs="+", required=True, type=Path,
                   help="judge output shards")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--stem", default="reprobe_onpolicy")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15,
                   help="Held out from both fitting and model selection, so an "
                        "in-domain test number means something. The archived run "
                        "had no such split and reported its selection set.")
    p.add_argument("--min_steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout_outcomes", type=Path, default=None,
                   help="the frozen-transfer evaluation outcomes, to re-verify "
                        "that no evaluation problem id leaked into training.")
    p.add_argument("--holdout_traces", nargs="*", type=Path, default=None,
                   help="the frozen-transfer evaluation trajectories, which carry "
                        "the question text. Ids are not enough: the same question "
                        "can hold two ids and pass the id check.")
    p.add_argument("--drop_holdout_questions", action="store_true",
                   help="Remove training-pool traces whose question text also "
                        "appears in the evaluation pool, instead of only counting "
                        "them. Conservative: it can only shrink the training set.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    traces = read_jsonl(args.traces)
    labels: dict[str, dict] = {}
    for f in args.labels:
        for r in read_jsonl(f):
            uid = r.get("traj_uid") or r.get("id")
            if uid is not None:
                labels[uid] = r
    joined, tally = join_labels(traces, labels, args.min_steps)
    if not joined:
        raise SystemExit("no labelled traces survived the join")

    holdout = {}
    if args.holdout_traces:
        held_q = {canonical_question(r["problem"])
                  for f in args.holdout_traces for r in read_jsonl(f)}
        shared = {canonical_question(t["problem"]) for t in joined} & held_q
        holdout["evaluation_questions"] = len(held_q)
        holdout["shared_question_texts"] = len(shared)
        if shared and args.drop_holdout_questions:
            before = len(joined)
            joined = [t for t in joined
                      if canonical_question(t["problem"]) not in shared]
            holdout["dropped_traces"] = before - len(joined)
            if not joined:
                raise SystemExit("dropping evaluation questions emptied the pool")
        elif shared:
            # Not fatal: the historical pool has this overlap and its results are
            # kept as a record. Reporting it is what stops the next run from
            # calling an id-disjoint split question-disjoint.
            print(f"[split] WARNING {len(shared)} question texts are shared with "
                  f"the evaluation pool; pass --drop_holdout_questions to remove "
                  f"them, or report every number fitted here as contaminated")

    train, val, test, info = split_by_question(
        joined, args.val_frac, args.test_frac, args.seed)
    sides = {name: {canonical_question(t["problem"]) for t in rows}
             for name, rows in (("train", train), ("val", val), ("test", test))}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (sides[a] & sides[b]), f"{a} and {b} share a question text"

    if args.holdout_outcomes and args.holdout_outcomes.exists():
        held = {r["problem_id"] for r in read_jsonl(args.holdout_outcomes)}
        leak = held & {t["problem_id"] for t in joined}
        if leak:
            raise SystemExit(
                f"{len(leak)} evaluation problems appear in the training pool; "
                f"the held-out set must stay untouched")
        holdout["evaluation_ids_checked"] = len(held)
    info["holdout"] = holdout

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        f = args.out_dir / f"{args.stem}_{name}.jsonl"
        if f.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {f}; pass --force")
        write_jsonl(f, rows)

    steps_tr = sum(len(t["steps"]) for t in train)
    faulty_tr = sum(len(t["faulty_steps"]) for t in train)
    manifest = {
        "counts": dict(sorted(tally.items())),
        "n_train_traces": len(train), "n_val_traces": len(val),
        "n_test_traces": len(test),
        "n_train_steps": steps_tr, "n_val_steps": sum(len(t["steps"]) for t in val),
        "n_test_steps": sum(len(t["steps"]) for t in test),
        "train_faulty_step_rate": faulty_tr / max(1, steps_tr),
        "train_traces_with_a_faulty_step":
            sum(1 for t in train if t["faulty_steps"]) / max(1, len(train)),
        "split": info, "traces": str(args.traces),
        "labels": [str(f) for f in args.labels],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }
    (args.out_dir / f"{args.stem}_split_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    for k, v in manifest["counts"].items():
        print(f"[split] {k:26s} {v}")
    print(f"[split] train {len(train)} traces / {steps_tr} steps over "
          f"{info['n_train_questions']} questions")
    print(f"[split] val   {len(val)} traces over {info['n_val_questions']} questions")
    print(f"[split] test  {len(test)} traces over {info['n_test_questions']} questions")
    print(f"[split] {info['n_questions']} questions carry {info['n_problem_ids']} "
          f"problem ids")
    print(f"[split] faulty-step rate in train {manifest['train_faulty_step_rate']:.3f}, "
          f"traces with any faulty step {manifest['train_traces_with_a_faulty_step']:.3f}")


if __name__ == "__main__":
    main()
