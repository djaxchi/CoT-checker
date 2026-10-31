import numpy as np

from scripts.reprobe_cmp.bon_analyze import pick, trace_scores


def test_trace_score_is_the_worst_step():
    sc = {"t1::0": 0.1, "t1::1": 0.7, "t2::0": 0.2}
    assert trace_scores(sc) == {"t1": 0.7, "t2": 0.2}


def test_pick_lowest_score_first_on_ties():
    qi = np.array([0, 0, 0, 1, 1])
    score = np.array([0.5, 0.2, 0.2, 0.9, 0.1])
    assert pick(qi, score) == {0: 1, 1: 4}


def test_within_auc():
    from scripts.reprobe_cmp.bon_analyze import within_auc
    qi = np.array([0, 0, 0, 1, 1, 2, 2])
    y = np.array([1, 0, 0, 1, 0, 1, 1])
    score = np.array([0.1, 0.5, 0.1, 0.9, 0.2, 0.3, 0.4])
    # q0: correct 0.1 vs wrong {0.5, 0.1} -> (1 + 0.5) / 2; q1: 0; q2 has no wrong one
    assert abs(within_auc(qi, y, score) - (0.75 + 0.0) / 2) < 1e-9
