import numpy as np

from scripts.analysis.online_compute_sim import (
    N_TRAJ, group_badness, prune, risk_coverage, simulate_a1, simulate_a2, vote)


def toy_cell():
    # one problem: trajectories 0-5 answer key 0 (correct), 6-9 answer key 1 (wrong)
    keys = np.array([[0] * 6 + [1] * 4])
    key_correct = np.zeros((1, N_TRAJ), bool)
    key_correct[0, 0] = True
    return {"keys": keys, "key_correct": key_correct, "tokens": np.full((1, N_TRAJ), 100.0),
            "correct": keys == 0}


def test_vote_breaks_ties_to_the_answer_drawn_first():
    keys = np.array([[[1, 0, 1, 0] + [0] * 6]])
    use = np.zeros((1, 1, N_TRAJ), bool)
    use[0, 0, :4] = True
    kc = np.zeros((1, N_TRAJ), bool)
    kc[0, 0] = True
    ok, anyu = vote(keys, use, kc)
    assert not ok[0, 0] and anyu[0, 0]  # 2 - 2 tie, key 1 drawn first, wrong
    use[0, 0, 0] = False
    ok, _ = vote(keys, use, kc)
    assert ok[0, 0]


def test_prune_kills_at_the_first_step_over_tau():
    m = np.maximum.accumulate(np.array([0.1, 0.5, 0.2, 0.9]))
    worst = [[(m, np.array([10.0, 20, 30, 40]))] + [(np.zeros(1), np.array([5.0]))] * (N_TRAJ - 1)]
    comp, cost = prune(worst, np.full((1, N_TRAJ), 50.0), 0.4)
    assert not comp[0, 0] and cost[0, 0] == 20 and comp[0, 1] and cost[0, 1] == 50


def test_a1_without_pruning_is_majority_and_oracle_saves_tokens():
    cell = toy_cell()
    perm = np.argsort(np.random.default_rng(0).random((1, 30, N_TRAJ)), -1)
    full = np.ones((1, N_TRAJ), bool)
    acc, tok = simulate_a1(cell, full, cell["tokens"], N_TRAJ, perm)
    assert acc[0] == 1.0 and tok[0] == 1000
    acc1, tok1 = simulate_a1(cell, full, cell["tokens"], 1, perm)
    assert 0 < acc1[0] < 1 and tok1[0] == 100
    comp = cell["correct"].copy()
    cost = np.where(comp, 100.0, 10.0)
    acc_o, tok_o = simulate_a1(cell, comp, cost, 1, perm)
    assert acc_o[0] == 1.0 and tok_o[0] < 200


def test_a2_stops_early_on_agreement():
    cell = toy_cell()
    cell["keys"] = np.zeros((1, N_TRAJ), np.int64)
    perm = np.argsort(np.random.default_rng(0).random((1, 5, N_TRAJ)), -1)
    acc, tok = simulate_a2(cell, np.ones((1, N_TRAJ), bool), cell["tokens"], 0.9, perm)
    assert acc[0] == 1.0 and tok[0] == 200


def test_group_badness_and_risk_coverage():
    b = group_badness(np.array([1.0, 3.0, 5.0]), 2, 1.0)
    assert np.allclose(b, [-1.0, -2.0, -4.0])
    rc = risk_coverage(np.array([0.9, 0.8, 0.1, 0.2]), np.array([True, True, False, False]))
    assert rc["acc_at_50"] == 1.0 and rc["acc_full"] == 0.5
