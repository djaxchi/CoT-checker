#!/usr/bin/env python3
"""Paired comparison of the online_reject_v4 arms on MATH-500 (REPORT §21.35).

Loads plain, reject_blind and reject rollouts from a run directory, keeps the
problems every arm finished, and reports per arm accuracy, generated tokens and
extra draws per step, plus paired differences with a problem bootstrap.
Usage: online_v4_control.py RUNS_DIR [--v3 V3_RUNS_DIR]
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np


def load(root: str, arm: str) -> dict:
    # A resumed job can redo a problem; keep the first rollout written.
    out = {}
    for f in sorted(glob.glob(f"{root}/{arm}/math500.shard*.jsonl")):
        for r in map(json.loads, open(f)):
            out.setdefault(r["problem_id"], r)
    return out


def extra_draws(r: dict) -> float:
    a = r["attempts_per_step"]
    return (sum(a) - len(a)) / max(len(a), 1)


def boot(d: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs")
    p.add_argument("--v3", default=None, help="v3 runs dir; adds its reject_boundary_stats_q50 arm")
    a = p.parse_args()
    arms = {k: load(a.runs, k) for k in ("plain", "reject_blind", "reject")}
    if a.v3:
        arms["reject_v3"] = load(a.v3, "reject_boundary_stats_q50")
        arms["plain_v3"] = load(a.v3, "plain")
    ids = sorted(set.intersection(*(set(v) for v in arms.values())))
    out = {"n": len(ids), "arms": {}, "diffs": {}}
    acc = {}
    for k, R in arms.items():
        acc[k] = np.array([float(R[i]["correct"]) for i in ids])
        out["arms"][k] = {"n_done": len(R), "acc": float(acc[k].mean()),
                          "tokens": float(np.mean([R[i].get("gen_tokens") or 0 for i in ids])),
                          "extra_draws_per_step": float(np.mean([extra_draws(R[i]) for i in ids
                                                                 if "attempts_per_step" in R[i]] or [0.0])),
                          "steps": float(np.mean([R[i]["n_steps"] for i in ids]))}
    pairs = [("reject", "reject_blind"), ("reject_blind", "plain"), ("reject", "plain")]
    if a.v3:
        pairs += [("reject_v3", "reject_blind"), ("plain", "plain_v3")]
    for x, y in pairs:
        d = acc[x] - acc[y]
        out["diffs"][f"{x} - {y}"] = {"mean_ci": boot(d), "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
