#!/usr/bin/env python3
"""Would adaptive rollout allocation have been safe? (docs/step_value_labels_v1_plan.md, section 2)

Replays a two-stage rule on factorized_step_labels_v1 full-run rollouts: draw 4
rollouts at both cuts of a fork (A: its first 4 continuations; C: the first
continuation of 4 different alternatives, i.e. 4 free rollouts from cut s) and
stop when all 8 agree. Reports the stop rate and how many forks the full-precision
labels call harmful would have been stopped with risk 0.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track1_dir", type=Path, required=True)
    ap.add_argument("--k0", type=int, default=4)
    a = ap.parse_args()
    rows = [json.loads(l) for p in sorted(glob.glob(str(a.track1_dir / "track1.shard*.jsonl"))) for l in open(p)]
    stop, harm, inc = [], [], []
    for r in rows:
        c = r["conditions"]
        alts = [k for k in c if k != "A"]
        first = [not x["correct"] for x in c["A"][:a.k0]] + [not c[j][0]["correct"] for j in alts[:a.k0]]
        stop.append(len(set(first)) == 1)
        rs = r["risk"]
        harm.append(rs["risk_diff_se"] > 0 and rs["risk_diff"] > 2 * rs["risk_diff_se"])
        inc.append(r["validity"] == "incorrect")
    stop, harm, inc = map(np.array, (stop, harm, inc))
    print(json.dumps({"n_forks": len(rows), "stop_rate": float(stop.mean()),
                      "stop_rate_incorrect": float(stop[inc].mean()), "stop_rate_correct": float(stop[~inc].mean()),
                      "harmful": int(harm.sum()), "harmful_stopped": int((stop & harm).sum())}, indent=1))


if __name__ == "__main__":
    main()
