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
