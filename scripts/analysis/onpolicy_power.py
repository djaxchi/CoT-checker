#!/usr/bin/env python3
"""How big does Phase 2 have to be to settle what Phase 1 could not?

Phase 1 left the verifier and answer-token-margin separated by -0.0308 with an
interval of [-0.1061, +0.0462] on 65 ties: a point estimate and no power. The
question that decides whether to spend the compute is not "is the effect real"
but "can a study of the planned size detect an effect of the observed size".

This computes that from the measured per-tie spread rather than an assumed one,
so the answer is anchored on this pool's own variance.

It also computes the number the scaling argument usually skips: what the
tie-break difference is worth in *final accuracy*, which is the tie-break
difference times the tie rate. A difference that needs ten thousand problems to
detect and moves final accuracy by one point is a different proposition from
the one the plan was written to chase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

Z_ALPHA = 1.959964      # two-sided 0.05
Z_POWER = 0.8416        # 80%

# Tie rates measured on the clean pool, REPORT.md §20.14 budget frontier.
MEASURED_TIE_RATE = {2: 0.744, 3: 0.560, 4: 0.453, 5: 0.379, 6: 0.326, 10: 0.244}


def per_tie_sd(delta_ci: tuple[float, float], n_ties: int) -> float:
    """Per-tie SD of the paired difference, recovered from a reported interval."""
    lo, hi = delta_ci
    return (hi - lo) / (2 * Z_ALPHA) * np.sqrt(n_ties)


def ties_needed(true_delta: float, sd: float) -> float:
    """Ties required for 80% power at two-sided 0.05."""
    if true_delta <= 0:
        return float("inf")
    return ((Z_ALPHA + Z_POWER) * sd / true_delta) ** 2


def detectable(n_ties: float, sd: float) -> float:
    """Smallest difference detectable at 80% power with this many ties."""
    return (Z_ALPHA + Z_POWER) * sd / np.sqrt(n_ties)


def final_accuracy_effect(tie_delta: float, tie_rate: float) -> float:
    """What a tie-break difference is worth end to end.

    A rule only acts where the vote ties, so its contribution to the number a
    user sees is the tie-break difference scaled by how often ties happen. This
    is the quantity the deployment argument rests on and the one a tie-only
    table hides.
    """
    return tie_delta * tie_rate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--confound_json", type=Path,
                   default=Path("results/onpolicy_format_confound_notrunc/format_confound.json"))
    p.add_argument("--contrast", type=str,
                   default="verifier_worst_step - answer_margin")
    p.add_argument("--out", type=Path, default=Path("results/onpolicy_power/power.json"))
    args = p.parse_args()

    r = json.loads(args.confound_json.read_text())
    pair = r["paired"][args.contrast]
    n = r["n_ties"]
    sd = per_tie_sd(tuple(pair["ci95"]), n)
    obs = abs(pair["delta"])

    # The three-set core of §3.2, with per-set tie rates at N=4 scaled from the
    # measured 0.453 at this pool's pass@1 of 0.375. GSM8K sits near 0.90, so it
    # contributes problems generously and ties barely at all.
    core = {"GSM8K": (1319, 0.12), "MATH500": (500, 0.38), "PRM800K test": (2000, 0.45)}
    per_half = {k: v * t * 0.5 for k, (v, t) in core.items()}
    total_half = sum(per_half.values())

    out = {
        "observed": {"contrast": args.contrast, "delta": pair["delta"],
                     "ci95": pair["ci95"], "n_ties": n,
                     "implied_per_tie_sd": float(sd)},
        "ties_needed_80pct": {str(d): float(ties_needed(d, sd))
                              for d in (0.02, 0.03, 0.05, 0.08, 0.10)},
        "core_three_set": {k: {"problems": core[k][0], "tie_rate_n4": core[k][1],
                               "ties_per_half": float(v)} for k, v in per_half.items()},
        "core_ties_per_half": float(total_half),
        "core_detectable_delta": float(detectable(total_half, sd)),
        "observed_delta_to_beat": float(obs),
        "core_is_powered": bool(detectable(total_half, sd) < obs),
        "final_accuracy_effect_at_observed_delta": {
            str(k): float(final_accuracy_effect(obs, v))
            for k, v in MEASURED_TIE_RATE.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
