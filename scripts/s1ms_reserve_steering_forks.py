#!/usr/bin/env python3
"""Reserve a small held-out set of PRM800K forks for the steering experiment.

Each reserved fork keeps the shared context plus both rated siblings, so the
steering harness can run the model on the prefix and test whether nudging the
representation flips the next step from the incorrect to the correct sibling.

Output JSONL rows: {fork_id, problem, prefix, positive_step, negative_step}.

Usage:
    python scripts/s1ms_reserve_steering_forks.py \\
        --items $FORKS/forks_val_items.jsonl --n_forks 15 --seed 7 \\
        --out runs/s1_model_size_dense/steering/steering_forks.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--n_forks", type=int, default=15)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.items.exists():
        raise SystemExit(f"[reserve] items not found: {args.items}")
    rows = [json.loads(l) for l in args.items.read_text().splitlines() if l.strip()]
    by_fork: dict = defaultdict(dict)
    for r in rows:
        by_fork[r["fork_id"]][r["role"]] = r

    usable = [fid for fid, roles in by_fork.items() if "positive" in roles and "negative" in roles]
    usable.sort()
    if not usable:
        raise SystemExit("[reserve] no forks with both a positive and a negative sibling")
    sel = usable if len(usable) <= args.n_forks else [
        usable[i] for i in np.random.default_rng(args.seed).choice(len(usable), size=args.n_forks, replace=False)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for fid in sel:
            pos, neg = by_fork[fid]["positive"], by_fork[fid]["negative"]
            f.write(json.dumps({
                "fork_id": fid,
                "problem": pos["problem"],
                "prefix": pos["prefix"],
                "positive_step": pos["candidate_step"],
                "negative_step": neg["candidate_step"],
            }, ensure_ascii=False) + "\n")
    print(f"[reserve] wrote {len(sel)} steering forks -> {args.out}")


if __name__ == "__main__":
    main()
