"""Length-stratified AUROC: the control has to collapse, real signal has to survive."""
import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from stratified_auroc import stratified_auroc  # noqa: E402


def _pb(n=8000, seed=0):
    """ProcessBench's actual shape: first-error steps run 118.6 tokens, the rest 79.7."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(np.int64)
    ln = np.where(y == 1, rng.normal(118.6, 40, n), rng.normal(79.7, 30, n))
    return y, np.maximum(ln, 5)


def test_length_collapses_to_chance_inside_its_own_bins():
    """And only at fine bins: coarse bins leave length usable, which is why the
    script prints this control rather than assuming a bin count is enough."""
    y, ln = _pb()
    got = {k: stratified_auroc(y, ln.astype(float), ln, k) for k in (1, 5, 10, 50)}
    assert got[1] > 0.75                       # unstratified, length is worth a lot
    assert got[5] > got[10] > got[50]          # shrinks monotonically with bin count
    assert got[50] == pytest.approx(0.5, abs=0.03)


def test_a_length_only_score_also_collapses():
    """A probe that learned nothing but length must lose everything here."""
    y, ln = _pb()
    s = 3.1 * np.log(ln) - 12.0
    assert stratified_auroc(y, s, ln, 50) == pytest.approx(0.5, abs=0.03)


def test_signal_independent_of_length_survives():
    y, ln = _pb()
    rng = np.random.default_rng(1)
    s = y + rng.normal(0, 1.0, len(y))          # informative, uncorrelated with length
    plain = stratified_auroc(y, s, ln, 1)
    assert stratified_auroc(y, s, ln, 50) == pytest.approx(plain, abs=0.02)
    assert plain > 0.7


def test_bins_with_one_class_are_dropped_not_counted_as_half():
    y = np.array([1, 1, 0, 0]); ln = np.array([1.0, 2.0, 3.0, 4.0])
    # bins [1,1] and [0,0] carry no comparable pair; only a 1-bin split has any
    assert np.isnan(stratified_auroc(y, np.array([1.0, 1.0, 0.0, 0.0]), ln, 2))
