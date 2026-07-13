"""Unit tests for the parametric_retrieval_access_v1 causal-experiment core."""

import numpy as np
import pandas as pd

from src.analysis.parametric_retrieval_causal import (
    PATCH_CONDITIONS,
    assign_patch_donors,
    budget_pairs,
    confound_features,
    estimate_directions,
    fact_bootstrap_ci,
    lda_direction,
    paired_diffs,
    residualize,
)


def make_pairs_and_groups(n_facts: int = 6):
    pairs, groups = [], []
    for i in range(n_facts):
        fid = f"f{i}"
        groups.append({"fact_id": fid, "direction": "direct",
                       "category": "Arts" if i % 2 else "Science",
                       "answer_type": "ORG" if i % 3 else "PERSON",
                       "gbc_bin": "low" if i % 2 else "high",
                       "donor_pool_instance_id": f"{fid}::succ0"})
        pairs.append({"pair_id": f"{fid}::p0", "fact_id": fid,
                      "direction": "direct",
                      "donor_instance_id": f"{fid}::succ0",
                      "recipient_instance_id": f"{fid}::fail0"})
    return pd.DataFrame(pairs), pd.DataFrame(groups)


def test_assign_patch_donors_conditions():
    pairs, groups = make_pairs_and_groups()
    out = assign_patch_donors(pairs, groups, seed=0)
    out2 = assign_patch_donors(pairs, groups, seed=0)
    pd.testing.assert_frame_equal(out, out2)  # deterministic
    for r in out.itertuples():
        assert r.donor_matched == r.donor_instance_id
        assert r.donor_noop == r.recipient_instance_id
        assert r.donor_reverse == r.recipient_instance_id
        # mismatched donors never come from the same fact
        assert not str(r.donor_mismatched_type).startswith(r.fact_id + ":")
        assert not str(r.donor_mismatched_rand).startswith(r.fact_id + ":")
    assert set(PATCH_CONDITIONS) == {"noop", "matched", "mismatched_type",
                                     "mismatched_rand", "random_noise",
                                     "reverse"}


def test_budget_pairs_keeps_fact_coverage():
    pairs = pd.DataFrame([
        {"pair_id": f"f{i}::p{j}", "fact_id": f"f{i}", "direction": "direct",
         "donor_instance_id": "d", "recipient_instance_id": "r"}
        for i in range(10) for j in range(8)])
    sub = budget_pairs(pairs, max_pairs=20, seed=1)
    assert len(sub) <= 20
    assert sub.fact_id.nunique() >= 3
    pd.testing.assert_frame_equal(sub, budget_pairs(pairs, 20, seed=1))
    assert len(budget_pairs(pairs, 1000)) == 80  # under budget: unchanged


def test_residualize_removes_linear_component():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((500, 3))
    B = rng.standard_normal((3, 8))
    X = F @ B + 0.01 * rng.standard_normal((500, 8))
    R = residualize(X, F)
    corr = np.abs(np.corrcoef(np.c_[R[:, 0], F].T)[0, 1:])
    assert corr.max() < 0.05
    assert np.abs(R).mean() < 0.1


def test_confound_features_shape():
    meta = pd.DataFrame({
        "template_id": ["w0", "w1", "w0"], "seed_variant": ["a", "a", "b"],
        "direction": ["direct"] * 3, "gbc_bin": ["low", "mid", "low"],
        "category": ["Arts"] * 3, "prompt_token_count": [40, 50, 60]})
    F = confound_features(meta)
    assert F.shape[0] == 3 and F.shape[1] >= 6
    assert np.isfinite(F).all()


def test_paired_diffs():
    H = np.array([[1.0, 0], [3.0, 0], [0, 1.0], [0, 3.0], [5.0, 5.0]])
    meta = pd.DataFrame({
        "fact_id": ["a", "a", "a", "a", "b"],
        "direction": ["direct"] * 5,
        "is_correct": [True, True, False, False, True]})
    diffs, keys = paired_diffs(H, meta)
    assert len(diffs) == 1  # fact b has no failures
    np.testing.assert_allclose(diffs[0], [2.0, -2.0])
    assert keys.fact_id.tolist() == ["a"]


def test_estimate_directions_unit_norm_and_alignment():
    rng = np.random.default_rng(0)
    true = np.zeros(16)
    true[0] = 1.0
    diffs = true + 0.1 * rng.standard_normal((200, 16))
    dirs = estimate_directions(diffs, seed=0, n_random=2)
    assert set(dirs) == {"mean_diff", "svd1", "svd4_proj",
                         "random_0", "random_1"}
    for v in dirs.values():
        assert abs(np.linalg.norm(v) - 1) < 1e-6
    assert abs(dirs["mean_diff"][0]) > 0.9
    assert abs(dirs["mean_diff"] @ dirs["svd4_proj"]) > 0.9
    assert abs(dirs["mean_diff"] @ dirs["random_0"]) < 0.6


def test_lda_direction_separates():
    rng = np.random.default_rng(1)
    H = rng.standard_normal((400, 10))
    y = rng.random(400) > 0.5
    H[y, 3] += 2.0
    w = lda_direction(H, y)
    assert abs(np.linalg.norm(w) - 1) < 1e-6
    assert abs(w[3]) > 0.7


def test_fact_bootstrap_ci():
    v = pd.Series([1.0] * 50 + [3.0] * 50)
    f = pd.Series([f"a{i}" for i in range(50)] + [f"b{i}" for i in range(50)])
    mean, lo, hi = fact_bootstrap_ci(v, f, n_boot=500, seed=0)
    assert abs(mean - 2.0) < 1e-9
    assert lo < 2.0 < hi
    assert hi - lo < 1.0
