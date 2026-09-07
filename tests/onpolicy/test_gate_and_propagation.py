"""Tests for the Phase 9 gate and its propagation diagnostic.

The diagnostics exist to tell two failure shapes apart: a head that localises an
error, and a head that merely notices a trace has gone on too long. So the tests
build both shapes explicitly and assert the metrics separate them. If these ever
stop discriminating, the Phase 9 conclusion loses its support.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.onpolicy_gate import paired_bootstrap, step_auroc_outcome
from scripts.analysis.onpolicy_propagation_check import (
    peak_concentration,
    position_coupling,
    suffix_ramp,
)


def _groups(traces):
    """traces: list of (problem, scores, correct) -> the grouped shape the analyses take."""
    out = {}
    for pid, scores, correct in traces:
        out.setdefault(pid, []).append({"scores": list(scores), "correct": correct})
    return out


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------

def test_paired_bootstrap_is_centred_on_zero_when_both_arms_are_identical():
    g = _groups([(f"p{i}", [0.1, 0.5, 0.9], i % 2 == 0) for i in range(40)])
    stat = lambda gg: float(np.mean([max(s["scores"]) for v in gg.values() for s in v]))
    mean, lo, hi = paired_bootstrap(g, g, stat, n_boot=200)
    assert mean == pytest.approx(0.0, abs=1e-12)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.0, abs=1e-12)


def test_paired_bootstrap_recovers_a_known_shift():
    a = _groups([(f"p{i}", [0.2], i % 2 == 0) for i in range(60)])
    b = _groups([(f"p{i}", [0.5], i % 2 == 0) for i in range(60)])
    stat = lambda gg: float(np.mean([max(s["scores"]) for v in gg.values() for s in v]))
    mean, lo, hi = paired_bootstrap(a, b, stat, n_boot=200)
    assert mean == pytest.approx(0.3, abs=1e-9)
    assert lo <= 0.3 + 1e-9 and hi >= 0.3 - 1e-9


def test_step_auroc_outcome_is_oriented_so_higher_means_more_suspicious():
    # failing traces carry uniformly higher step scores than passing ones
    g = _groups(
        [(f"p{i}", [0.8, 0.9], False) for i in range(20)]
        + [(f"q{i}", [0.1, 0.2], True) for i in range(20)]
    )
    assert step_auroc_outcome(g) > 0.99

    flipped = _groups(
        [(f"p{i}", [0.1, 0.2], False) for i in range(20)]
        + [(f"q{i}", [0.8, 0.9], True) for i in range(20)]
    )
    assert step_auroc_outcome(flipped) < 0.01


# --------------------------------------------------------------------------
# propagation diagnostic: the two shapes it must tell apart
# --------------------------------------------------------------------------

LOCALISED = [0.05, 0.05, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05]   # one bad step
LATENESS = [0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]    # whole suffix bad


def test_position_coupling_separates_a_spike_from_a_ramp():
    spike = _groups([(f"p{i}", LOCALISED, False) for i in range(30)])
    ramp = _groups([(f"p{i}", LATENESS, False) for i in range(30)])
    rho_spike = position_coupling(spike)["mean_spearman_score_vs_position"]
    rho_ramp = position_coupling(ramp)["mean_spearman_score_vs_position"]
    assert rho_ramp > rho_spike
    assert rho_ramp > 0.5, "a suffix of bad steps must read as strongly position-coupled"


def test_position_coupling_ignores_correct_traces_by_default():
    mixed = _groups(
        [(f"p{i}", LATENESS, False) for i in range(10)]
        + [(f"q{i}", LOCALISED, True) for i in range(10)]
    )
    assert position_coupling(mixed)["n_traces"] == 10


def test_peak_concentration_is_one_for_a_flat_trace_and_n_for_a_single_spike():
    flat = _groups([(f"p{i}", [0.4] * 8, False) for i in range(10)])
    assert peak_concentration(flat)["peakiness_x_flat"] == pytest.approx(1.0, abs=1e-9)

    spike = _groups([(f"p{i}", [0.0] * 7 + [1.0], False) for i in range(10)])
    assert peak_concentration(spike)["peakiness_x_flat"] == pytest.approx(8.0, abs=1e-9)


def test_peak_concentration_ranks_localised_above_propagated():
    spike = _groups([(f"p{i}", LOCALISED, False) for i in range(10)])
    ramp = _groups([(f"p{i}", LATENESS, False) for i in range(10)])
    assert (peak_concentration(spike)["peakiness_x_flat"]
            > peak_concentration(ramp)["peakiness_x_flat"])


def test_suffix_ramp_rises_only_for_the_propagated_shape():
    # error at step 5 of 9, so each third is genuinely a different mix
    late = [0.05] * 5 + [0.95] * 4
    ramp = suffix_ramp(_groups([(f"p{i}", late, False) for i in range(10)]))
    assert ramp["first_third"] < ramp["middle_third"] < ramp["last_third"]

    # an error at a fixed early position does not produce a rising staircase
    early = suffix_ramp(_groups([(f"p{i}", [0.95, 0.95, 0.05, 0.05, 0.05, 0.05], False)
                                 for i in range(10)]))
    assert early["first_third"] > early["last_third"]
