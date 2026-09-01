#!/usr/bin/env python3
"""Write {cell: off-policy F1_PB at calib-20} for the downstream correlation.

Recomputed from each cell's saved per-trace ProcessBench scores under the same
protocol the leaderboard uses, imported rather than restated, so the x-axis of
the T2 correlation is the published number and not a near-miss of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from merge_rep_grid_leaderboard import CALIB_SIZE, PB_SUBSETS, calib20_subset, load_traces  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid_root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    per_cell: dict[str, list[float]] = defaultdict(list)
    for d in sorted(args.grid_root.iterdir()):
        rj = d / "results.json"
        if not rj.exists():
            continue
        res = json.loads(rj.read_text())
        vals = []
        for sub in PB_SUBSETS:
            f = d / f"pb_step_scores_{sub}.jsonl"
            if not f.exists():
                break
            tr = load_traces(f)
            if len(tr) <= CALIB_SIZE:
                break
            vals.append(calib20_subset(tr))
        if len(vals) == len(PB_SUBSETS):
            per_cell[f"{res['rep']} x {res['learner']}"].append(float(np.mean(vals)))

    out = {k: float(np.mean(v)) for k, v in per_cell.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]):
        print(f"{k:<48}{v:.4f}  ({len(per_cell[k])} seeds)")
    print(f"[offpolicy] wrote {args.out} ({len(out)} cells)")


if __name__ == "__main__":
    main()
