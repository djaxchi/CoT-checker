"""transition_operator_v0: freeze problem-disjoint train/val/test splits to JSON.

House rule: splits are materialized BEFORE any extraction/training job; all jobs
read the frozen file. Grouping is by question (problem-disjoint), 80/10/10.

  python scripts/transition_operator/to_splits.py --run_dir runs/transition_operator
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = args.run_dir / "splits.json"
    if out.exists() and not args.force:
        sys.exit(f"refusing to overwrite frozen {out}; pass --force")

    forks = [json.loads(l) for l in open(args.run_dir / "forks.jsonl") if l.strip()]
    by_q: dict[str, list[str]] = {}
    for fk in forks:
        by_q.setdefault(fk["question"], []).append(fk["fork_id"])
    questions = sorted(by_q)
    random.Random(args.seed).shuffle(questions)
    n = len(questions)
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    split_q = {"test": questions[:n_test],
               "val": questions[n_test:n_test + n_val],
               "train": questions[n_test + n_val:]}
    splits = {name: sorted(fid for q in qs for fid in by_q[q])
              for name, qs in split_q.items()}
    out.write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_questions": {k: len(v) for k, v in split_q.items()},
        "n_forks": {k: len(v) for k, v in splits.items()},
        **splits}, indent=2))
    print(f"[splits] {out}: " + ", ".join(
        f"{k} {len(v)} forks / {len(split_q[k])} questions" for k, v in splits.items()))


if __name__ == "__main__":
    main()
