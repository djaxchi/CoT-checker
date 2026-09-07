#!/usr/bin/env python3
"""Did the on-policy head learn to localise errors, or just to notice lateness?

The GPT-OSS labels mark the entire suffix after the first error 55.6% of the
time. A head trained on that can score well without localising anything: it only
has to learn that steps late in a doomed trace are bad. Phase 9 showed the
step-level gain (+0.083) running ~2.7x the within-problem gain (+0.031), which is
the shape that pattern would produce, so this tests it directly.

Three independent probes of the same question, because any one of them alone has
an innocent explanation:

  position coupling   how much of a step's score is predicted by *where* it sits
                      in the trace. A localiser spikes at one step; a lateness
                      detector ramps monotonically.

  aggregation profile if the score means "this trace has gone wrong by now",
                      then the LAST step already carries the verdict and
                      last-step aggregation should rival worst-step. For a true
                      localiser the worst step is informative and the last step
                      is not special.

  peak concentration  how much of a failing trace's total suspicion sits in its
                      single highest step. Localisation concentrates; propagation
                      spreads.

None of these needs step labels, which the held-out pool does not have.
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
    cell_solutions,
    load_outcomes,
)


def position_coupling(groups: dict[str, list[dict]], only_incorrect: bool = True) -> dict:
    """Spearman(step score, normalised step position) within each trace, averaged.

    Computed inside a trace, so trace-level difficulty cannot drive it.
    """
    rhos, slopes = [], []
    for sols in groups.values():
        for sol in sols:
            if only_incorrect and sol["correct"]:
                continue
            s = np.asarray(sol["scores"], dtype=float)
            n = s.size
            if n < 4 or np.allclose(s, s[0]):
                continue
            pos = np.arange(n, dtype=float) / (n - 1)
            rs = _rank(s)
            rp = _rank(pos)
            rho = float(np.corrcoef(rs, rp)[0, 1])
            if np.isfinite(rho):
                rhos.append(rho)
            # least-squares slope of score on normalised position
            slopes.append(float(np.polyfit(pos, s, 1)[0]))
    return {
        "n_traces": len(rhos),
        "mean_spearman_score_vs_position": float(np.mean(rhos)) if rhos else float("nan"),
        "share_positive_rho": float(np.mean([r > 0 for r in rhos])) if rhos else float("nan"),
        "mean_slope_over_trace": float(np.mean(slopes)) if slopes else float("nan"),
    }


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(x.size, dtype=float)
    return r


def peak_concentration(groups: dict[str, list[dict]]) -> dict:
    """Share of a failing trace's total suspicion carried by its top step.

    Normalised by 1/n so a flat trace scores 1.0 regardless of length; a trace
    whose suspicion sits entirely in one step scores n.
    """
    vals, tops = [], []
    for sols in groups.values():
        for sol in sols:
            if sol["correct"]:
                continue
            s = np.asarray(sol["scores"], dtype=float)
            n = s.size
            if n < 4 or s.sum() <= 0:
                continue
            share = s.max() / s.sum()
            vals.append(share * n)          # 1.0 = perfectly flat
            tops.append(float(share))
    return {
        "n_traces": len(vals),
        "peakiness_x_flat": float(np.mean(vals)) if vals else float("nan"),
        "mean_top_step_share": float(np.mean(tops)) if tops else float("nan"),
    }


def suffix_ramp(groups: dict[str, list[dict]]) -> dict:
    """Mean score of the first, middle and last third of each failing trace.

    A propagating label set produces a rising staircase; a localiser produces a
    bump wherever the error is, which averages flat across traces.
    """
    thirds = [[], [], []]
    for sols in groups.values():
        for sol in sols:
            if sol["correct"]:
                continue
            s = np.asarray(sol["scores"], dtype=float)
            if s.size < 6:
                continue
            k = s.size // 3
            thirds[0].append(s[:k].mean())
            thirds[1].append(s[k:2 * k].mean())
            thirds[2].append(s[2 * k:].mean())
    return {
        "first_third": float(np.mean(thirds[0])) if thirds[0] else float("nan"),
        "middle_third": float(np.mean(thirds[1])) if thirds[1] else float("nan"),
        "last_third": float(np.mean(thirds[2])) if thirds[2] else float("nan"),
        "n_traces": len(thirds[0]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--offpolicy_grid", required=True, type=Path)
    p.add_argument("--offpolicy_scores", default="onpolicy_verifier")
    p.add_argument("--onpolicy_grid", required=True, type=Path)
    p.add_argument("--onpolicy_scores", default="verifier")
    p.add_argument("--outcomes", required=True, type=Path)
    p.add_argument("--cells", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    outcomes = load_outcomes(a.outcomes)
    report = {"cells": []}

    for cell in a.cells:
        fa = a.offpolicy_grid / cell / f"pb_step_scores_{a.offpolicy_scores}.jsonl"
        fb = a.onpolicy_grid / cell / f"pb_step_scores_{a.onpolicy_scores}.jsonl"
        if not (fa.exists() and fb.exists()):
            continue
        ga, gb = cell_solutions(fa, outcomes), cell_solutions(fb, outcomes)
        entry = {
            "cell": cell,
            "offpolicy": {
                "position": position_coupling(ga),
                "peak": peak_concentration(ga),
                "ramp": suffix_ramp(ga),
            },
            "onpolicy": {
                "position": position_coupling(gb),
                "peak": peak_concentration(gb),
                "ramp": suffix_ramp(gb),
            },
        }
        report["cells"].append(entry)

        o, n = entry["offpolicy"], entry["onpolicy"]
        print(f"=== {cell}")
        print(f"  score-vs-position rho   off {o['position']['mean_spearman_score_vs_position']:+.3f}"
              f"   on {n['position']['mean_spearman_score_vs_position']:+.3f}"
              f"   delta {n['position']['mean_spearman_score_vs_position'] - o['position']['mean_spearman_score_vs_position']:+.3f}")
        print(f"  share of traces rho>0   off {o['position']['share_positive_rho']:.3f}"
              f"   on {n['position']['share_positive_rho']:.3f}")
        print(f"  peakiness (1.0 = flat)  off {o['peak']['peakiness_x_flat']:.3f}"
              f"   on {n['peak']['peakiness_x_flat']:.3f}")
        print(f"  ramp first/mid/last     off {o['ramp']['first_third']:.3f}/{o['ramp']['middle_third']:.3f}/{o['ramp']['last_third']:.3f}"
              f"   on {n['ramp']['first_third']:.3f}/{n['ramp']['middle_third']:.3f}/{n['ramp']['last_third']:.3f}")
        print()

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
        print(f"[propagation] wrote {a.out}")


if __name__ == "__main__":
    main()
