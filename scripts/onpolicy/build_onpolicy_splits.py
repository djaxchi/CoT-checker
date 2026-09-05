#!/usr/bin/env python3
"""Split the labelled on-policy pool into train and validation, by problem.

Splitting by trajectory would put ten samples of the same problem on both sides
and let the probe memorise the problem rather than learn what a faulty step looks
like. The split is therefore over problem ids, deterministic under a seed, and
the assignment is written out so a later run can prove it used the same one.

The 300 held-out evaluation problems are not in this pool at all (verified at
generation time and re-verified here), so this splits train from validation only.
The test set is the frozen-transfer evaluation pool and is never touched.

Emits ProcessBench-shaped traces carrying `step_labels`, which the encoder reads
to mark every faulty step rather than only a first error.
"""

from __future__ import annotations

import argparse
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


def split_by_problem(traces: list[dict], val_frac: float, seed: int
                     ) -> tuple[list[dict], list[dict], dict]:
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
    p.add_argument("--min_steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout_outcomes", type=Path, default=None,
                   help="the frozen-transfer evaluation outcomes, to re-verify "
                        "that no evaluation problem leaked into training.")
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

    train, val, info = split_by_problem(joined, args.val_frac, args.seed)
    assert not ({t["problem_id"] for t in train} & {t["problem_id"] for t in val}), \
        "train and validation share a problem"

    if args.holdout_outcomes and args.holdout_outcomes.exists():
        held = {r["problem_id"] for r in read_jsonl(args.holdout_outcomes)}
        leak = held & {t["problem_id"] for t in joined}
        if leak:
            raise SystemExit(
                f"{len(leak)} evaluation problems appear in the training pool; "
                f"the held-out set must stay untouched")
        info["holdout_checked"] = len(held)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        f = args.out_dir / f"{args.stem}_{name}.jsonl"
        if f.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {f}; pass --force")
        write_jsonl(f, rows)

    steps_tr = sum(len(t["steps"]) for t in train)
    faulty_tr = sum(len(t["faulty_steps"]) for t in train)
    manifest = {
        "counts": dict(sorted(tally.items())),
        "n_train_traces": len(train), "n_val_traces": len(val),
        "n_train_steps": steps_tr, "n_val_steps": sum(len(t["steps"]) for t in val),
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
          f"{info['n_problems'] - info['n_val_problems']} problems")
    print(f"[split] val   {len(val)} traces over {info['n_val_problems']} problems")
    print(f"[split] faulty-step rate in train {manifest['train_faulty_step_rate']:.3f}, "
          f"traces with any faulty step {manifest['train_traces_with_a_faulty_step']:.3f}")


if __name__ == "__main__":
    main()
