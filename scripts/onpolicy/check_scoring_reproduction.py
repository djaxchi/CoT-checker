#!/usr/bin/env python3
"""Does the downstream scoring path reproduce each checkpoint's own scores?

Every cell wrote `pb_step_scores_gsm8k.jsonl` at training time. Rescoring the
same ProcessBench split through `score_cells_on_split.py` must give the same
per-step scores, or the downstream numbers come from a different function than
the one the leaderboard ranked (lengthfree_geom is the known risk: its
training-fitted length transform is not applied on that path). A cell whose
maximum absolute difference exceeds the tolerance is marked failed and must not
enter the downstream ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, list[float]]:
    return {r["id"]: [float(x) for x in r["scores"]]
            for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def compare(ref: dict, new: dict) -> float:
    if set(ref) != set(new):
        return float("inf")
    worst = 0.0
    for k, a in ref.items():
        b = new[k]
        if len(a) != len(b):
            return float("inf")
        worst = max([worst] + [abs(x - y) for x, y in zip(a, b)])
    return worst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells_root", type=Path, required=True, help="the trained cells")
    p.add_argument("--rescored_root", type=Path, required=True)
    p.add_argument("--split", default="gsm8k")
    p.add_argument("--tol", type=float, default=2e-3)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    rows = []
    for f in sorted(a.rescored_root.glob(f"*/pb_step_scores_{a.split}.jsonl")):
        cell = f.parent.name
        ref = a.cells_root / cell / f"pb_step_scores_{a.split}.jsonl"
        d = compare(load(ref), load(f)) if ref.exists() else float("inf")
        rows.append({"cell": cell, "max_abs_diff": d, "pass": d <= a.tol})
        print(f"[repro] {'PASS' if d <= a.tol else 'FAIL'} {cell:<52} max|diff| {d:.2e}")
    a.out.write_text(json.dumps({"tol": a.tol, "split": a.split, "cells": rows}, indent=1))
    n_pass = sum(r["pass"] for r in rows)
    print(f"[repro] {n_pass}/{len(rows)} cells reproduce their training-time scores")


if __name__ == "__main__":
    main()
