#!/usr/bin/env python3
"""ProcessBench F1_PB with an oracle threshold, next to calib-20, for stored per-trace step scores.

Each --arm name=dir[,dir...] points at one directory per seed holding pb_step_scores_<subset>.jsonl
(analyze.write_pb's format, the files paper_v1.py scores calib-20 from). Per subset and seed:
  calib-20  merge_rep_grid_leaderboard.calib20_subset (the paper's number, as a check)
  oracle    the best F1_PB over every candidate threshold on the whole subset: the uniform grid
            0.01..0.99 that the leaderboard's oracle used, plus 999 quantiles of the subset's
            scores (overconfident heads put most scores near 0 or 1, where the uniform grid is
            coarse); the same candidates for every arm
Seed means, then the 4-subset mean. Writes <out> (json) and prints a table.

    python scripts/instep_v1/pb_oracle.py --out $OUT/final/pb_oracle.json \\
        --arm instep_ds=$OUT/final/pb_scores/qwen3_8b/instep_ds/seed0,... prm7b=...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.pb_threshold_calibration import load_traces  # noqa: E402
from scripts.merge_rep_grid_leaderboard import (CALIB_GRID, PB_SUBSETS, calib20_subset,  # noqa: E402
                                                f1_pb_from_preds, pred_matrix)


def oracle_subset(traces: list) -> tuple[float, float]:
    """(best F1_PB, its threshold) over the uniform grid and the score quantiles."""
    s = np.concatenate([np.asarray(sc, dtype=np.float64) for _, sc in traces])
    grid = np.unique(np.concatenate([CALIB_GRID, np.quantile(s, np.linspace(0.0005, 0.9995, 999))]))
    preds, labels = pred_matrix(traces, grid)
    f1 = f1_pb_from_preds(preds, labels)
    i = int(np.argmax(f1))
    return float(f1[i]), float(grid[i])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", nargs="+", required=True, help="name=dir[,dir...] (one dir per seed)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    res = {}
    for spec in a.arm:
        name, dirs = spec.split("=", 1)
        per = {"calib20": {s: [] for s in PB_SUBSETS}, "oracle": {s: [] for s in PB_SUBSETS}}
        for d in dirs.split(","):
            for s in PB_SUBSETS:
                tr = load_traces(Path(d) / f"pb_step_scores_{s}.jsonl")
                per["calib20"][s].append(calib20_subset(tr))
                per["oracle"][s].append(oracle_subset(tr)[0])
        r = {m: {s: float(np.mean(v)) for s, v in per[m].items()} for m in per}
        for m in per:
            r[m]["mean"] = float(np.mean([r[m][s] for s in PB_SUBSETS]))
        r["seeds"] = len(dirs.split(","))
        res[name] = r
        print(f"{name:12s} seeds {r['seeds']} | calib-20 {r['calib20']['mean']:.3f} | oracle {r['oracle']['mean']:.3f} ("
              + " / ".join(f"{r['oracle'][s]:.3f}" for s in PB_SUBSETS) + ")", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
