#!/usr/bin/env python3
"""Phase 9: does training the verifier on the policy's own trajectories help?

The frozen-transfer arm asked whether a PRM800K-trained verifier is useful on
Qwen's own output. This asks the paired follow-up: hold the representation, the
architecture, the backbone, the eval pool and the aggregation rule fixed, change
only *what the head was trained on*, and measure the difference on the same
2,873 held-out trajectories.

Both arms score identical trajectory ids in identical order, so every comparison
here is paired at the problem level. That matters more than usual: the pool spans
four ProcessBench subsets of very different difficulty, and an unpaired
difference between two verifiers is mostly a statement about which problems
happened to land where.

Three questions, deliberately kept apart:

  1. does on-policy training beat off-policy training      (the gate)
  2. does either beat self-consistency                     (the frozen arm's finding)
  3. do step-level gains show up downstream                (the propagation check)

Question 3 is the one that can embarrass us. The GPT-OSS labels mark the whole
suffix after the first error 55.6% of the time, so a head trained on them can
score well at step level by learning "this trace has gone wrong by now" rather
than "this step is the wrong one". If step AUROC moves and within-problem AUROC
and best-of-N do not, that is the shortcut and not a better verifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.analysis.onpolicy_downstream import (  # noqa: E402
    aggregate,
    auroc,
    best_of_n_hits,
    cell_solutions,
    load_outcomes,
    mcnemar,
    self_consistency_hits,
    within_problem_auroc,
)


def paired_bootstrap(
    groups_a: dict[str, list[dict]],
    groups_b: dict[str, list[dict]],
    stat,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample *problems*, not solutions, and recompute the paired delta.

    Solutions inside one problem are not independent (same question, same
    policy, often the same answer), so bootstrapping over solutions would
    understate the interval badly.
    """
    keys = sorted(set(groups_a) & set(groups_b))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sub = [keys[j] for j in pick]
        ga = {f"{k}#{j}": groups_a[k] for j, k in enumerate(sub)}
        gb = {f"{k}#{j}": groups_b[k] for j, k in enumerate(sub)}
        deltas[i] = stat(gb) - stat(ga)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def step_auroc_outcome(groups: dict[str, list[dict]]) -> float:
    """Label-free step metric: do steps of failing solutions score higher?

    The held-out pool carries no per-step ground truth (label == -1 throughout),
    so this is the shadow of a step metric, not F1_PB. Named accordingly.
    """
    s, y = [], []
    for sols in groups.values():
        for sol in sols:
            for v in sol["scores"]:
                s.append(v)
                y.append(0 if sol["correct"] else 1)
    return auroc(np.asarray(y), np.asarray(s))


def cell_metrics(groups: dict[str, list[dict]], how: str) -> dict:
    return {
        "best_of_n": float(np.mean(list(best_of_n_hits(groups, how).values()))),
        "within_problem_auroc": within_problem_auroc(groups, how),
        "traj_auroc": _traj_auroc(groups, how),
        "step_auroc_outcome": step_auroc_outcome(groups),
    }


def _traj_auroc(groups: dict[str, list[dict]], how: str) -> float:
    s, y = [], []
    for sols in groups.values():
        for sol in sols:
            s.append(aggregate(sol["scores"], how))
            y.append(0 if sol["correct"] else 1)
    return auroc(np.asarray(y), np.asarray(s))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--offpolicy_grid", required=True, type=Path)
    p.add_argument("--offpolicy_scores", default="onpolicy_verifier")
    p.add_argument("--onpolicy_grid", required=True, type=Path)
    p.add_argument("--onpolicy_scores", default="verifier")
    p.add_argument("--outcomes", required=True, type=Path)
    p.add_argument("--cells", nargs="+", required=True,
                   help="cell directory names present in BOTH grids")
    p.add_argument("--how", default="worst_step", choices=["worst_step", "mean_step", "last_step"])
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    outcomes = load_outcomes(a.outcomes)
    report = {"aggregation": a.how, "n_boot": a.n_boot, "cells": []}

    # self-consistency, computed once, identical for both arms
    any_cell = a.cells[0]
    ref = cell_solutions(a.onpolicy_grid / any_cell / f"pb_step_scores_{a.onpolicy_scores}.jsonl", outcomes)
    sc = self_consistency_hits(ref)
    report["self_consistency"] = float(np.mean(list(sc.values())))
    report["n_problems"] = len(ref)
    report["n_solutions"] = sum(len(v) for v in ref.values())

    print(f"{report['n_problems']} problems, {report['n_solutions']} solutions, "
          f"aggregation = {a.how}")
    print(f"self-consistency {report['self_consistency']:.3f}\n")

    for cell in a.cells:
        fa = a.offpolicy_grid / cell / f"pb_step_scores_{a.offpolicy_scores}.jsonl"
        fb = a.onpolicy_grid / cell / f"pb_step_scores_{a.onpolicy_scores}.jsonl"
        if not fa.exists() or not fb.exists():
            print(f"SKIP {cell}: missing {'off' if not fa.exists() else 'on'}-policy scores")
            continue
        ga = cell_solutions(fa, outcomes)
        gb = cell_solutions(fb, outcomes)
        shared = sorted(set(ga) & set(gb))
        assert len(shared) == len(ga) == len(gb), (
            f"{cell}: pools differ ({len(ga)} off vs {len(gb)} on, {len(shared)} shared); "
            "the comparison would not be paired"
        )

        off = cell_metrics(ga, a.how)
        on = cell_metrics(gb, a.how)
        delta = {k: on[k] - off[k] for k in off}

        # paired significance on the decision that matters
        hits_off = best_of_n_hits(ga, a.how)
        hits_on = best_of_n_hits(gb, a.how)
        mc = mcnemar(hits_off, hits_on)
        boot_mean, lo, hi = paired_bootstrap(
            ga, gb,
            lambda g: float(np.mean(list(best_of_n_hits(g, a.how).values()))),
            n_boot=a.n_boot,
        )
        # and against the real baseline
        mc_vs_sc = mcnemar(sc, hits_on)

        entry = {
            "cell": cell,
            "offpolicy": off,
            "onpolicy": on,
            "delta": delta,
            "bon_delta_mcnemar": mc,
            "bon_delta_bootstrap": {"mean": boot_mean, "lo95": lo, "hi95": hi},
            "onpolicy_vs_self_consistency": {
                "delta": on["best_of_n"] - report["self_consistency"],
                "mcnemar": mc_vs_sc,
            },
        }
        report["cells"].append(entry)

        print(f"=== {cell}")
        for k in ("best_of_n", "within_problem_auroc", "traj_auroc", "step_auroc_outcome"):
            print(f"  {k:24s} off {off[k]:.3f}   on {on[k]:.3f}   delta {delta[k]:+.3f}")
        print(f"  best-of-N delta paired bootstrap {boot_mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"  McNemar p={mc['p']:.4g} (off wins {mc.get('a_wins',0)}, on wins {mc.get('b_wins',0)}, discordant {mc['n_discordant']})")
        print(f"  on-policy vs self-consistency    {entry['onpolicy_vs_self_consistency']['delta']:+.3f}"
              f"  McNemar p={mc_vs_sc['p']:.4g}")
        print()

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
        print(f"[gate] wrote {a.out}")


if __name__ == "__main__":
    main()
