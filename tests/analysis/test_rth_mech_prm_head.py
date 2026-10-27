import json

import numpy as np
import torch

from scripts.analysis import rth_mech_prm_head as m5


def test_cv_ridge_finds_the_row_that_carries_the_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 2, 10))
    y = X[:, 1] @ rng.normal(size=10) + 0.1 * rng.normal(size=1200)
    folds = m5.group_folds([f"t{i // 4}" for i in range(1200)], 5)
    good = m5.cv_ridge(X[:, 1], y, folds, torch.device("cpu"))
    bad = m5.cv_ridge(X[:, 0], y, folds, torch.device("cpu"))
    assert good["r2"] > 0.99 and bad["r2"] < 0.02
    assert abs(good["r2"] - (1 - ((good["oof"] - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())) < 1e-9


def test_folds_keep_traces_together():
    groups = [f"t{i // 3}" for i in range(300)]
    f = m5.group_folds(groups, 5)
    for g in set(groups):
        assert len({int(f[i]) for i, x in enumerate(groups) if x == g}) == 1
    assert len(set(f.tolist())) == 5


def test_disagreement_counts_sides():
    rth = np.array([1.0, 1.0, -1.0, -1.0, 2.0])
    prm = np.array([-1.0, 1.0, 1.0, -1.0, -2.0])
    cls = np.array([1, 1, 0, 0, -1])   # last step is after the first error: unlabelled
    pred = np.array([-0.5, 0.0, 0.5, 0.0, 9.0])
    d = m5.disagreement(rth, prm, cls, {"p": pred}, "p05")
    assert d["n_disagree"] == 2 and d["n_labelled"] == 4
    assert d["rth_right"] == 1.0 and d["prm_right"] == 0.0     # RTH is right on both
    assert d["rth_flags_prm_not"] == 1 and d["prm_flags_rth_not"] == 1
    assert d["pred_sides_with_prm"]["p"] == 1.0


def test_prm_targets_follow_bundle_order(tmp_path):
    rows = [{"id": "a", "log_odds_incorrect": [0.5, -1.0]}, {"id": "b", "log_odds_incorrect": [2.0]}]
    (tmp_path / "pb_step_scores_math.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = [{"id": "b", "step_idx": 0}, {"id": "a", "step_idx": 1}, {"id": "a", "step_idx": 0}]
    assert m5.prm_targets(tmp_path, meta, "math").tolist() == [2.0, -1.0, 0.5]


def test_rate_threshold_matches_the_error_rate():
    s = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 9.0])
    cls = np.array([1, 0, 0, 0, 1, -1])     # 2 of 5 labelled steps are first errors
    v = m5.verdicts(s, cls, "rate")
    assert v[:5].tolist() == [True, True, False, False, False]
