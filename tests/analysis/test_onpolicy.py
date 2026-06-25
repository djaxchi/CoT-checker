"""Unit tests for the Stage 1 on-policy generation + analysis helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("generate_onpolicy_steps", "generate_onpolicy_steps.py")
ana = _load("analyze_onpolicy_probe", "analyze_onpolicy_probe.py")


# --------------------------------------------------------------------------- #
# step splitting + item building
# --------------------------------------------------------------------------- #

def test_split_on_blank_lines():
    sol = "Step one reasoning.\n\nStep two reasoning.\n\nThus \\boxed{4}."
    steps = gen.split_into_steps(sol)
    assert steps == ["Step one reasoning.", "Step two reasoning.", "Thus \\boxed{4}."]


def test_split_falls_back_to_single_newline():
    sol = "line a\nline b\nline c"
    assert gen.split_into_steps(sol) == ["line a", "line b", "line c"]
    assert gen.split_into_steps("   ") == []


def test_build_step_items_prefix_and_label():
    items = gen.build_step_items("P", "4", "a\n\nb\n\nc", "traj1", traj_correct=False)
    assert [it["candidate_step"] for it in items] == ["a", "b", "c"]
    assert items[0]["prefix"] == ""               # first step has empty prefix
    assert items[1]["prefix"] == "a"              # PRM800K \n\n join
    assert items[2]["prefix"] == "a\n\nb"
    assert all(it["label"] == 1 for it in items)  # incorrect trajectory -> label 1
    assert all(it["role"] == "generated" for it in items)
    assert items[2]["item_uid"] == "traj1::step2" and items[2]["fork_id"] == "traj1"


def test_build_step_items_correct_traj_label_zero():
    items = gen.build_step_items("P", "4", "x\n\ny", "t", traj_correct=True)
    assert all(it["label"] == 0 and it["traj_correct"] for it in items)


def test_unique_problems_dedupes_and_requires_gold():
    forks = [
        {"fork_id": "f1", "problem": "P1", "ground_truth_answer": "1", "role": "positive"},
        {"fork_id": "f1", "problem": "P1", "ground_truth_answer": "1", "role": "negative"},
        {"fork_id": "f2", "problem": "P2", "ground_truth_answer": "", "role": "positive"},
        {"fork_id": "f3", "problem": "P3", "ground_truth_answer": "9", "role": "positive"},
    ]
    probs = gen.unique_problems(forks)
    assert {p["fork_id"] for p in probs} == {"f1", "f3"}   # f1 deduped, f2 dropped (no gold)


# --------------------------------------------------------------------------- #
# analysis: bootstrap CI + trajectory aggregation
# --------------------------------------------------------------------------- #

def test_trajectory_scores_aggregates_by_fork():
    fork_ids = ["a", "a", "b", "b", "b"]
    score = np.array([1.0, 3.0, 0.0, 0.0, 6.0])
    label = np.array([1, 1, 0, 0, 0])
    ts, tl = ana.trajectory_scores(fork_ids, score, label)
    order = np.argsort(["a", "b"])  # deterministic mapping check
    assert set(np.round(ts, 3)) == {2.0, 2.0}      # mean(1,3)=2 ; mean(0,0,6)=2
    assert sorted(tl.tolist()) == [0, 1]


def test_bootstrap_ci_brackets_true_auc():
    rng = np.random.default_rng(0)
    n = 400
    label = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    score = np.r_[rng.normal(0, 1, n // 2), rng.normal(2, 1, n // 2)]
    point = ana.auc(score, label)
    lo, hi = ana.bootstrap_auc_ci(score, label, n_boot=500)
    assert lo < point < hi
    assert lo > 0.5            # clearly-separated signal stays above chance


# --------------------------------------------------------------------------- #
# F1 at a threshold (headline metric): oracle, val-selected, trivial baseline
# --------------------------------------------------------------------------- #

def test_oracle_f1_perfectly_separable_is_one():
    score = np.array([0.0, 0.1, 0.2, 0.9, 1.0, 1.1])
    label = np.array([0, 0, 0, 1, 1, 1])
    f1, thr, prec, rec = ana.oracle_f1(score, label)
    assert f1 == pytest.approx(1.0)
    assert prec == pytest.approx(1.0) and rec == pytest.approx(1.0)
    # threshold lands on the lowest positive score (rule is score >= thr)
    assert 0.2 < thr <= 0.9


def test_oracle_f1_matches_bruteforce():
    rng = np.random.default_rng(3)
    score = rng.normal(0, 1, 200)
    label = (rng.random(200) < 0.4).astype(int)
    f1_vec, _, _, _ = ana.oracle_f1(score, label)
    best = 0.0
    for t in np.unique(score):
        f1, _, _ = ana._f1_at(score, label, t)
        best = max(best, f1)
    assert f1_vec == pytest.approx(best, abs=1e-9)


def test_oracle_f1_beats_trivial_when_signal_present():
    # separable signal -> oracle F1 should clear the predict-all-positive strawman
    rng = np.random.default_rng(1)
    n = 300
    label = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    score = np.r_[rng.normal(0, 1, n // 2), rng.normal(3, 1, n // 2)]
    f1, _, _, _ = ana.oracle_f1(score, label)
    trivial = 2 * label.mean() / (1 + label.mean())
    assert f1 > trivial


def test_val_selected_f1_freezes_threshold_on_held_out_half():
    rng = np.random.default_rng(2)
    n = 400
    label = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    score = np.r_[rng.normal(0, 1, n // 2), rng.normal(3, 1, n // 2)]
    f1_test, thr, prec, rec, f1_val = ana.val_selected_f1(score, label)
    assert 0.0 <= f1_test <= 1.0 and 0.0 <= f1_val <= 1.0
    # honest (held-out) F1 should not exceed the oracle ceiling on the same data
    oracle, _, _, _ = ana.oracle_f1(score, label)
    assert f1_test <= oracle + 1e-9


def test_f1_helpers_handle_degenerate_single_class():
    score = np.array([0.1, 0.2, 0.3])
    label = np.array([0, 0, 0])
    f1, thr, p, r = ana.oracle_f1(score, label)
    assert np.isnan(f1)
