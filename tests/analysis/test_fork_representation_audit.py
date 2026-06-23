"""Unit tests for the pure-math helpers of the fork representation audit."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "fork_audit", ROOT / "scripts" / "analyze_fork_representation_audit.py"
)
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)


def test_build_triples_groups_and_drops():
    meta = [
        {"fork_id": "f1", "role": "anchor", "item_uid": "f1::a", "row": 0, "n_tokens": 5},
        {"fork_id": "f1", "role": "positive", "item_uid": "f1::p", "row": 1, "n_tokens": 10},
        {"fork_id": "f1", "role": "negative", "item_uid": "f1::n", "row": 2, "n_tokens": 12},
        {"fork_id": "f2", "role": "positive", "item_uid": "f2::p", "row": 3, "n_tokens": 7},
        # f2 has no negative -> dropped
        {"fork_id": "f3", "role": "positive", "item_uid": "f3::p", "row": 4, "n_tokens": 8},
        {"fork_id": "f3", "role": "positive", "item_uid": "f3::p2", "row": 5, "n_tokens": 9},
        {"fork_id": "f3", "role": "negative", "item_uid": "f3::n", "row": 6, "n_tokens": 11},
    ]
    triples, n_multi = fa.build_triples(meta)
    fids = {t["fork_id"] for t in triples}
    assert fids == {"f1", "f3"}                      # f2 dropped
    assert n_multi == 1                               # f3 had 2 positives
    t1 = next(t for t in triples if t["fork_id"] == "f1")
    assert (t1["pos_row"], t1["neg_row"], t1["anchor_row"]) == (1, 2, 0)
    t3 = next(t for t in triples if t["fork_id"] == "f3")
    assert t3["anchor_row"] is None and t3["pos_row"] == 4   # first positive kept


def test_probe_scores_and_cosine():
    H = np.array([[1.0, 0.0], [0.0, 2.0]])
    w = np.array([2.0, 1.0])
    s = fa.probe_scores(H, w, b=0.5)
    assert np.allclose(s, [2.5, 2.5])
    assert fa.cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert fa.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert fa.cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


def test_delta_stats_constant_shift():
    # neg = pos + constant vector -> effect size huge, sign consistency 1.
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(200, 3))
    shift = np.array([1.0, -2.0, 0.0])
    neg = pos + shift
    st = fa.delta_stats(pos, neg)
    assert np.allclose(st["mean_shift"], shift, atol=1e-9)
    assert st["sign_consistency"][0] == pytest.approx(1.0)
    assert st["sign_consistency"][1] == pytest.approx(1.0)
    # dim 0/1 have zero variance in delta -> infinite/nan effect; dim 2 shift 0
    assert st["mean_shift"][2] == pytest.approx(0.0)


def test_common_energy_bounds_and_meaning():
    # identical deltas -> all energy in the mean -> common_energy == 1.
    D = np.tile(np.array([1.0, 2.0, -1.0]), (50, 1))
    assert fa.common_energy(D) == pytest.approx(1.0)
    # zero-mean random deltas -> common energy near 0.
    rng = np.random.default_rng(1)
    Dr = rng.normal(size=(5000, 10))
    assert fa.common_energy(Dr) < 0.05


def test_sign_flip_null_kills_common_direction():
    rng = np.random.default_rng(2)
    D = np.tile(np.array([3.0, 0.0]), (400, 1)) + rng.normal(scale=0.1, size=(400, 2))
    true_energy = fa.common_energy(D)
    null = fa.sign_flip_null(D, n_iter=100, rng=rng)
    assert true_energy > null["energy_p95"]           # real shared direction beats null


def test_fake_unmatched_has_no_self_match():
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(50, 4))
    neg = pos + np.array([1.0, 0, 0, 0])
    Dfake = fa.fake_unmatched_deltas(pos, neg, rng)
    # a true matched delta would be exactly the shift; fake should differ for all rows
    assert not np.any(np.all(np.isclose(Dfake, np.array([1.0, 0, 0, 0])), axis=1))


def test_var_along_recovers_planted_direction():
    rng = np.random.default_rng(4)
    d = np.array([1.0, 0.0, 0.0])
    X = rng.normal(scale=0.01, size=(500, 3))
    X[:, 0] += rng.normal(scale=5.0, size=500)        # nearly all variance on axis 0
    assert fa.var_along(d, X) > 0.95
    assert fa.var_along(np.array([0.0, 1.0, 0.0]), X) < 0.05


def test_surface_features_detects_number_change():
    f = fa.surface_features("we get 2 + 2 = 4", "we get 2 + 2 = 5", n_tok_pos=6, n_tok_neg=6)
    assert f["length_diff"] == 0
    assert f["numbers_changed"] >= 1                  # 4 vs 5 differ
    assert 0.0 <= f["token_overlap"] <= 1.0


def test_auc_orientation():
    # neg (label-1) scored higher than pos (label-0) -> AUC > 0.5, not < 0.5.
    pos = np.array([0.0, 0.1, 0.2, 0.3])
    neg = np.array([0.7, 0.8, 0.9, 1.0])
    assert fa._auc(neg, pos) == pytest.approx(1.0)
    assert fa._auc(pos, neg) == pytest.approx(0.0)   # swapped args would invert


def test_surface_residualize_removes_length_component():
    # Plant a delta = a*length_diff (pure surface) + small true direction.
    rng = np.random.default_rng(7)
    n = 300
    ld = rng.normal(size=n)
    surf = [{"length_diff": ld[i], "char_dissim": 0.0, "token_overlap": 1.0,
             "number_diff": 0, "numbers_changed": 0, "operator_changed": 0} for i in range(n)]
    surf_dir = np.array([1.0, 0.0, 0.0])
    true_dir = np.array([0.0, 1.0, 0.0])
    D = ld[:, None] * surf_dir[None, :] + 0.05 * rng.normal(size=(n, 1)) * true_dir[None, :]
    X = fa.surface_design_matrix(surf)
    D_res = fa.surface_residualize(D, X)
    # the length-driven component on axis 0 should be largely gone
    assert np.var(D_res[:, 0]) < 0.1 * np.var(D[:, 0])


def test_surface_residualize_preserves_mean_when_surface_uncorrelated():
    # Surface uncorrelated with the margin -> residualization must NOT remove the
    # directional signal (guards against the intercept/mean-removal artifact).
    rng = np.random.default_rng(11)
    n, H = 400, 32
    w = np.zeros(H); w[0] = 1.0
    D = rng.normal(size=(n, H)); D[:, 0] += 0.7      # consistent shift along w
    surf = [{"length_diff": float(rng.normal()), "char_dissim": float(rng.normal()),
             "token_overlap": float(rng.normal()), "number_diff": float(rng.normal()),
             "numbers_changed": float(rng.normal()), "operator_changed": float(rng.normal()),
             "neg_has_final": 0, "pos_has_final": 0} for _ in range(n)]
    X = fa.surface_design_matrix(surf)
    D_res = fa.surface_residualize(D, X)
    p_raw = np.mean((D @ w) > 0)
    p_res = np.mean((D_res @ w) > 0)
    assert abs(p_res - p_raw) < 0.05                 # margin survives (was 0.5 with the bug)
    assert (D_res @ w).mean() > 0.4                  # mu_D preserved


def test_minimal_edit_mask_keeps_similar_pairs():
    surf = [{"token_overlap": ov, "length_diff": ld}
            for ov, ld in [(0.9, 1), (0.2, 50), (0.95, 0), (0.1, 80), (0.8, 2)]]
    mask = fa.minimal_edit_mask(surf, q=0.5)
    # high-overlap, low length-diff pairs kept; low-overlap high-diff dropped
    assert mask[2]            # ov 0.95, ld 0
    assert not mask[3]        # ov 0.10, ld 80


def test_trigger_rates_shape_and_sign():
    rng = np.random.default_rng(5)
    H_all = rng.normal(size=(100, 6))
    # sparse outliers (what |z|>2 trigger detection is for): a minority of negatives
    # spike dim 0 hard, so they sit in the pooled tail while positives do not.
    H_all[50:60, 0] += 50.0
    out = fa.trigger_rates(H_all, np.arange(50), np.arange(50, 100), z_thresh=2.0)
    assert out["diff_neg_minus_pos"].shape == (6,)
    assert out["diff_neg_minus_pos"][0] > 0           # dim 0 triggers more on negatives
