import numpy as np

from scripts.instep_v1.pb_oracle import bootstrap_pair, first_error, mean_point, oracle_subset, p05_subset
from scripts.merge_rep_grid_leaderboard import PB_SUBSETS, calib20_subset


def _traces(rng, n=200, noise=0.0):
    out = []
    for i in range(n):
        L = int(rng.integers(3, 9))
        s = rng.uniform(0, 0.3, L)
        if i % 2:  # error trace: first error at step k scores high
            k = int(rng.integers(0, L))
            s[k] = rng.uniform(0.6, 1.0)
            out.append((k, (s + rng.normal(0, noise, L)).tolist()))
        else:
            out.append((-1, (s + rng.normal(0, noise, L)).tolist()))
    return out


def test_oracle_is_perfect_on_separable_scores_and_bounds_calib20():
    rng = np.random.default_rng(0)
    tr = _traces(rng)
    f1, t = oracle_subset(tr)
    assert f1 == 1.0 and 0.29 <= t < 0.6  # scores > t flag a step
    assert calib20_subset(tr) <= f1 + 1e-12
    assert p05_subset(tr) == 1.0
    assert first_error(tr) == (1.0, 1.0)


def test_oracle_finds_thresholds_between_overconfident_scores():
    rng = np.random.default_rng(1)
    tr = [(lab, [1e-4 + 1e-6 * x for x in sc]) for lab, sc in _traces(rng)]  # all scores below 0.01
    assert oracle_subset(tr)[0] == 1.0
    assert p05_subset(tr) < 1.0


def test_bootstrap_pair_sign_and_zero_for_identical_arms():
    rng = np.random.default_rng(2)
    good = [{s: _traces(np.random.default_rng(10 + i)) for i, s in enumerate(PB_SUBSETS)}]
    bad = [{s: [(lab, (np.array(sc) + rng.normal(0, 0.4, len(sc))).tolist()) for lab, sc in good[0][s]]
            for s in PB_SUBSETS}]
    r = bootstrap_pair(good, bad, 30)
    assert r["auroc"]["delta"] > 0 and r["auroc"]["ci95"][0] > 0
    z = bootstrap_pair(good, good, 10)
    assert z["ap"]["delta"] == 0 and z["ap"]["ci95"] == [0.0, 0.0]
    assert set(mean_point(good)) == {"auroc", "ap", "oracle", "p05"}
