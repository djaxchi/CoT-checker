"""The hand-rolled statistics, checked against values that are known.

The p-value path is a continued-fraction incomplete beta written out because
scipy is not in the cluster's offline wheelhouse. A subtly wrong one would
produce plausible p-values for every correlation in the final table, so it is
checked against closed forms rather than eyeballed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import pytest  # noqa: E402

from scripts.analysis.onpolicy_transfer_report import _betainc, spearman_p  # noqa: E402


@pytest.mark.parametrize("a,b,x,want", [
    # I_x(1,1) = x, the uniform case
    (1.0, 1.0, 0.25, 0.25),
    (1.0, 1.0, 0.80, 0.80),
    # I_x(a,b) at x=1 is 1 and at x=0 is 0, for any a,b
    (2.5, 3.5, 1.0, 1.0),
    (2.5, 3.5, 0.0, 0.0),
    # I_x(2,1) = x^2
    (2.0, 1.0, 0.5, 0.25),
    # I_x(1,2) = 1-(1-x)^2
    (1.0, 2.0, 0.5, 0.75),
    # symmetry: I_0.5(a,a) = 0.5
    (3.0, 3.0, 0.5, 0.5),
])
def test_incomplete_beta_matches_closed_forms(a, b, x, want):
    assert _betainc(a, b, x) == pytest.approx(want, abs=1e-6)


def test_a_perfect_correlation_is_significant_and_a_null_one_is_not():
    assert spearman_p(0.99, 19) < 0.001
    assert spearman_p(0.0, 19) == pytest.approx(1.0, abs=1e-6)


def test_the_p_value_falls_as_the_correlation_rises():
    ps = [spearman_p(r, 19) for r in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert ps == sorted(ps, reverse=True)


def test_the_same_correlation_is_less_significant_with_fewer_cells():
    """n=6 is the representation-family unit and n=19 the cell unit; the same
    rho must not be reported as equally strong evidence in both."""
    assert spearman_p(0.7, 6) > spearman_p(0.7, 19)


def test_degenerate_inputs_do_not_produce_a_number():
    assert math.isnan(spearman_p(1.0, 19))     # |rho| = 1 has no t statistic
    assert math.isnan(spearman_p(0.5, 3))      # too few points
