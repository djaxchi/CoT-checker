"""Unit tests for the full-scale fork builder (pure functions, no model)."""

import random

from scripts.build_prm800k_forks import (
    build_split_artifacts,
    split_forks_by_problem,
    valid_forks,
)


def _ex(pid, sid, sidx, cidx, rating):
    label = 0 if rating == 1 else 1
    return {
        "uid": f"prm800k::{pid}::{sid}::{sidx}::{cidx}",
        "problem_id": pid,
        "solution_id": sid,
        "step_idx": sidx,
        "completion_idx": cidx,
        "problem": f"problem-{pid}",
        "ground_truth_answer": "42",
        "prefix": f"prefix-{pid}-{sidx}",
        "candidate_step": f"step-{cidx}",
        "rating": rating,
        "label": label,
    }


def _fork_map():
    # fork A: 1 pos, 2 neg ; fork B: 2 pos, 1 neg ; fork C: pos only (degenerate)
    return {
        "pA::sA::0": [_ex("pA", "sA", 0, 0, 1), _ex("pA", "sA", 0, 1, -1), _ex("pA", "sA", 0, 2, -1)],
        "pB::sB::1": [_ex("pB", "sB", 1, 0, 1), _ex("pB", "sB", 1, 1, 1), _ex("pB", "sB", 1, 2, -1)],
        "pC::sC::0": [_ex("pC", "sC", 0, 0, 1)],
    }


def test_valid_forks_drops_degenerate():
    forks = valid_forks(_fork_map())
    assert set(forks.keys()) == {"pA::sA::0", "pB::sB::1"}
    assert len(forks["pA::sA::0"]["neg"]) == 2


def test_split_is_problem_disjoint():
    forks = valid_forks(_fork_map())
    train, val = split_forks_by_problem(forks, n_train=1, n_val=1, rng=random.Random(0))
    train_pids = {k.split("::", 1)[0] for k in train}
    val_pids = {k.split("::", 1)[0] for k in val}
    assert train_pids.isdisjoint(val_pids)


def test_build_all_mode_emits_cartesian_pairs_and_anchor():
    forks = valid_forks(_fork_map())
    items, pairs, stats = build_split_artifacts(
        "train", ["pA::sA::0"], forks, pair_mode="all", rng=random.Random(0)
    )
    # fork A: 1 pos x 2 neg = 2 pairs
    assert len(pairs) == 2
    # items: 1 anchor + 1 positive + 2 negatives (deduped) = 4
    roles = sorted(i["role"] for i in items)
    assert roles == ["anchor", "negative", "negative", "positive"]
    # every pair carries the shared fork anchor
    assert all(pr["anchor_uid"].endswith("::anchor") for pr in pairs)
    assert stats["train_pairs_if_all"] == 2
    assert stats["train_pairs_if_one"] == 1


def test_build_one_mode_single_pair_per_fork():
    forks = valid_forks(_fork_map())
    items, pairs, stats = build_split_artifacts(
        "train", ["pA::sA::0", "pB::sB::1"], forks, pair_mode="one", rng=random.Random(1)
    )
    assert len(pairs) == 2  # one per fork
    # anchor label is sentinel -1, candidate_step empty
    anchors = [i for i in items if i["role"] == "anchor"]
    assert len(anchors) == 2
    assert all(a["label"] == -1 and a["candidate_step"] == "" for a in anchors)


def test_item_uids_unique():
    forks = valid_forks(_fork_map())
    items, _, _ = build_split_artifacts(
        "train", ["pA::sA::0", "pB::sB::1"], forks, pair_mode="all", rng=random.Random(2)
    )
    uids = [i["item_uid"] for i in items]
    assert len(uids) == len(set(uids))
