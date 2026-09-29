#!/usr/bin/env python3
"""Leaderboard of every trained cell on each backbone, and their rank agreement.

Reads the cells as they stand, so a cell with fewer than three finished seeds
enters with the seeds it has and the table says how many. For each
(representation, learner) and backbone it reports the seed mean of:

  auroc      in-domain AUROC on the PRM800K balanced test (results.json)
  val        ProcessBench F1_PB at the PRM800K-selected threshold, 4-subset mean
  oracle     ProcessBench F1_PB at the per-subset oracle threshold, 4-subset mean
  calib20    ProcessBench F1_PB with the threshold chosen on 20 traces of each
             subset, mean over 20 draws, 4-subset mean (the leaderboard metric)

and the Spearman correlation between the two backbones' cell orderings on each
metric, over the cells both backbones trained under the same protocol.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.pb_threshold_calibration import load_traces, sweep  # noqa: E402

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
GRID = np.arange(0.01, 1.0, 0.01)
METRICS = ("auroc", "val", "oracle", "calib20")


def cell_metrics(d: Path) -> dict:
    r = json.loads((d / "results.json").read_text())
    pb = r["processbench"]
    calib = []
    for s in SUBSETS:
        res = sweep(load_traces(d / f"pb_step_scores_{s}.jsonl"), [20], list(range(20)), GRID)
        calib.append(res["calibration_sweep"][0]["eval_f1_mean"])
    return {"rep": r["rep"], "learner": r["learner"], "seed": r["seed"],
            "rescale": r["protocol"].get("rescale") or "none(pre-field)",
            "auroc": r["in_domain"]["auroc"],
            "val": float(np.mean([pb[s]["val_selected"]["F1_PB"] for s in SUBSETS])),
            "oracle": float(np.mean([pb[s]["oracle_F1_PB"] for s in SUBSETS])),
            "calib20": float(np.mean(calib))}


def collect(roots: list[Path]) -> dict:
    by = defaultdict(list)
    for root in roots:
        for d in sorted(root.glob("*__seed*")):
            if d.is_dir() and (d / "results.json").exists() and not d.name.startswith("sae_"):
                m = cell_metrics(d)
                by[(m["rep"], m["learner"])].append(m)
    return by


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, nargs="+", required=True)
    p.add_argument("--instruct", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    grids = {"base": collect(a.base), "instruct": collect(a.instruct)}
    summary = {}
    for arm, by in grids.items():
        for key, ms in by.items():
            summary.setdefault(key, {})[arm] = {
                "n_seeds": len(ms), "seeds": sorted(m["seed"] for m in ms),
                "rescale": sorted({m["rescale"] for m in ms}),
                **{k: float(np.mean([m[k] for m in ms])) for k in METRICS},
                **{k + "_sd": float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else None
                   for k in METRICS}}

    # Same protocol on both sides: raw states. Base cells that were z-scored
    # (the later lengthfree grid) are listed but kept out of the correlation.
    raw = {"none", "none(pre-field)"}
    shared = sorted(k for k, v in summary.items()
                    if "base" in v and "instruct" in v
                    and set(v["base"]["rescale"]) <= raw and set(v["instruct"]["rescale"]) <= raw)
    rho = {m: spearman([summary[k]["base"][m] for k in shared],
                       [summary[k]["instruct"][m] for k in shared]) for m in METRICS}

    ranked = sorted(summary.items(), key=lambda kv: -kv[1].get("instruct", {}).get("calib20", -1))
    lines = ["| rank | representation | learner | seeds I | I AUROC | I val | I oracle | I calib20 "
             "| B AUROC | B calib20 | B rank |", "|" + "---|" * 11]
    base_rank = {k: i + 1 for i, (k, _) in enumerate(sorted(
        [kv for kv in summary.items() if "base" in kv[1]], key=lambda kv: -kv[1]["base"]["calib20"]))}
    for i, (k, v) in enumerate(ranked, 1):
        I, B = v.get("instruct"), v.get("base")
        f = lambda d, m: "-" if d is None else f"{d[m]:.4f}"
        lines.append(f"| {i} | {k[0]} | {k[1]} | {I['n_seeds'] if I else 0} | {f(I,'auroc')} | "
                     f"{f(I,'val')} | {f(I,'oracle')} | {f(I,'calib20')} | {f(B,'auroc')} | "
                     f"{f(B,'calib20')} | {base_rank.get(k, '-')}"
                     f"{'' if k in shared else ' (not in rho)'} |")
    lines += ["", f"Spearman, Base against Instruct, over {len(shared)} cells trained on raw "
              "states on both backbones: " + ", ".join(f"{m} {rho[m]:+.3f}" for m in METRICS)]
    md = "\n".join(lines)
    print(md)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".md").write_text(md + "\n")
    a.out.write_text(json.dumps({"spearman": rho, "shared_cells": [list(k) for k in shared],
                                 "cells": [{"rep": k[0], "learner": k[1], **v}
                                           for k, v in summary.items()]}, indent=1))


if __name__ == "__main__":
    main()
