"""Tests for the in-domain test-split carving in build_prm800k_full.

carve_disjoint_test must return a balanced test split that is problem-disjoint
from the remaining train pool, conserve all examples, and never mutate inputs.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_prm800k_full import carve_disjoint_test  # noqa: E402


def make_pool(n_problems: int, per_problem: int = 1) -> tuple[list[dict], list[dict]]:
    """One pos and one neg example per (problem_id, k), stable uids."""
    pos, neg = [], []
    for p in range(n_problems):
        for k in range(per_problem):
            pid = f"prob{p}"
            pos.append({"problem_id": pid, "uid": f"{pid}_pos{k}", "label": 0})
            neg.append({"problem_id": pid, "uid": f"{pid}_neg{k}", "label": 1})
    return pos, neg


def test_sizes_and_balance():
    pos, neg = make_pool(20)
    rng = random.Random(0)
    _, _, pos_test, neg_test = carve_disjoint_test(pos, neg, 3, 3, rng)
    assert len(pos_test) == 3
    assert len(neg_test) == 3


def test_problem_disjoint_from_train():
    pos, neg = make_pool(20)
    rng = random.Random(1)
    pos_tr, neg_tr, pos_test, neg_test = carve_disjoint_test(pos, neg, 4, 4, rng)
    test_pids = {e["problem_id"] for e in pos_test + neg_test}
    train_pids = {e["problem_id"] for e in pos_tr + neg_tr}
    assert test_pids & train_pids == set()


def test_conservation_no_loss():
    pos, neg = make_pool(15)
    rng = random.Random(2)
    pos_tr, neg_tr, pos_test, neg_test = carve_disjoint_test(pos, neg, 2, 2, rng)
    got = {e["uid"] for e in pos_tr + neg_tr + pos_test + neg_test}
    want = {e["uid"] for e in pos + neg}
    assert got == want


def test_inputs_not_mutated():
    pos, neg = make_pool(10)
    pos_before = [dict(e) for e in pos]
    neg_before = [dict(e) for e in neg]
    carve_disjoint_test(pos, neg, 2, 2, random.Random(3))
    assert pos == pos_before
    assert neg == neg_before


def test_zero_request_returns_copies():
    pos, neg = make_pool(5)
    pos_tr, neg_tr, pos_test, neg_test = carve_disjoint_test(pos, neg, 0, 0, random.Random(4))
    assert pos_test == [] and neg_test == []
    assert pos_tr == pos and pos_tr is not pos


def test_insufficient_pool_exits():
    pos, neg = make_pool(2)
    with pytest.raises(SystemExit):
        carve_disjoint_test(pos, neg, 5, 5, random.Random(5))
