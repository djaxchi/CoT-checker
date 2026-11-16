import numpy as np

from scripts.instep_v1.pb_oracle import oracle_subset
from scripts.merge_rep_grid_leaderboard import calib20_subset


def _traces(rng, n=200):
    out = []
    for i in range(n):
        L = int(rng.integers(3, 9))
        s = rng.uniform(0, 0.3, L)
        if i % 2:  # error trace: first error at step k scores high
            k = int(rng.integers(0, L))
            s[k] = rng.uniform(0.6, 1.0)
            out.append((k, s.tolist()))
        else:
            out.append((-1, s.tolist()))
    return out


def test_oracle_is_perfect_on_separable_scores_and_bounds_calib20():
    rng = np.random.default_rng(0)
    tr = _traces(rng)
    f1, t = oracle_subset(tr)
    assert f1 == 1.0 and 0.3 <= t < 0.6
    assert calib20_subset(tr) <= f1 + 1e-12


def test_oracle_finds_thresholds_between_overconfident_scores():
    rng = np.random.default_rng(1)
    tr = [(lab, [1e-4 + 1e-6 * x for x in sc]) for lab, sc in _traces(rng)]  # all scores below 0.01
    assert oracle_subset(tr)[0] == 1.0
