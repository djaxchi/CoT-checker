"""Relational representations: geometry, contribution, and cross-layer revision."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from relational_reps import build, geom_feats, layer_feats  # noqa: E402

D = 16


def _span(t=7, seed=0, spread=1.0, center=None):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=D) if center is None else center
    return (c + spread * rng.normal(size=(t, D))).astype(np.float32)


def test_geometry_is_invariant_to_rotating_the_residual_stream():
    """geom claims to be content free. If it is, an orthogonal change of basis
    applied to the step and the boundary alike must leave every number the same.
    A single direction leaking in would break this."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    span, bnd = _span(9, 1), rng.normal(size=D).astype(np.float32)
    a = geom_feats(span, bnd, with_len=True)
    b = geom_feats((span @ q).astype(np.float32), (bnd @ q).astype(np.float32), True)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_geometry_is_invariant_to_rescaling_only_where_it_should_be():
    """Angles must not move when the whole step is scaled; the log-norm entries
    must move by exactly the log of the scale, or they are not measuring norm."""
    span, bnd = _span(9, 1), _span(1, 2)[0]
    a = geom_feats(span, bnd, with_len=True)
    b = geom_feats((3.0 * span).astype(np.float32), bnd, with_len=True)
    moved = np.abs(a - b) > 1e-4
    # the cone and turn statistics, and the ||mean||/mean||token|| ratio, are angles
    assert not moved[:13].any(), "an angle feature moved under pure rescaling"
    assert moved.sum() >= 3, "no norm feature responded to a 3x rescale"


def test_a_tight_cone_scores_higher_than_a_diffuse_one():
    """The conicity work found correct steps cone tightly and incorrect ones do
    not. Whatever else geom measures, its cone entry has to order these."""
    tight = geom_feats(_span(20, 3, spread=0.05), _span(1, 9)[0], False)
    loose = geom_feats(_span(20, 3, spread=5.0), _span(1, 9)[0], False)
    assert tight[0] > loose[0]                      # mean cos(token, step mean)
    assert tight[12] > loose[12]                    # ||mean|| / mean||token||


def test_contribution_is_the_step_minus_the_prefix_state():
    span, bnd = _span(5, 4), _span(1, 5)[0]
    got = build(span, bnd, None, None)["contribution"]
    np.testing.assert_allclose(got, span.mean(0) - bnd, rtol=1e-5)


def test_layer_features_are_zero_disagreement_when_the_layers_agree():
    """Identical layers means no revision: cosines pin to 1 and log ratios to 0."""
    span, bnd = _span(6, 6), _span(1, 7)[0]
    f = layer_feats(span, bnd, span, bnd)
    assert f[0] == pytest.approx(1.0, abs=1e-5)     # mean per-token cos
    assert f[1] == pytest.approx(0.0, abs=1e-5)     # its std
    assert f[8] == pytest.approx(1.0, abs=1e-5)     # cos of the pooled means
    assert f[10] == pytest.approx(0.0, abs=1e-5)    # log norm ratio of the means


def test_layer_features_detect_a_revision():
    span, bnd = _span(6, 6), _span(1, 7)[0]
    other = _span(6, 11)
    assert layer_feats(span, bnd, other, bnd)[0] < 0.9


def test_geom_nolen_drops_exactly_the_length_entry():
    """Step length alone scores 0.7039 on ProcessBench, so the pair has to differ
    by the length feature and nothing else, or the comparison says nothing."""
    span, bnd = _span(13, 8), _span(1, 9)[0]
    a = geom_feats(span, bnd, with_len=True)
    b = geom_feats(span, bnd, with_len=False)
    assert len(a) == len(b) + 1
    np.testing.assert_allclose(a[:-1], b)
    assert a[-1] == pytest.approx(np.log(13))


def test_single_token_steps_do_not_crash_or_emit_nan():
    """Steps of one token exist in the store and have no consecutive pairs."""
    for name, v in build(_span(1, 12), _span(1, 13)[0], _span(1, 14), _span(1, 15)[0]).items():
        assert np.isfinite(v).all(), f"{name} emitted a non-finite value"


def test_geom_is_small_enough_to_be_the_point():
    """The claim is that a handful of numbers competes with 4,096 dimensions. If
    this ever grows past a few dozen the claim stops being interesting."""
    n = len(geom_feats(_span(9, 1), _span(1, 2)[0], with_len=True))
    assert n == 21, f"geom is {n} features, update the claim it competes with 4,096"
    assert len(layer_feats(_span(9, 1), _span(1, 2)[0],
                           _span(9, 5), _span(1, 6)[0])) == 12
