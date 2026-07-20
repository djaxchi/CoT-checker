"""Unit tests for progress_usefulness_v0 P0 pair builder (pure functions, no model).

Covers: the keep_neutral retention path in build_candidates (backward compatible
by default), the +1/0 fork filter, the shared-prefix invariant of emitted pairs,
progress_label mapping, and problem-disjoint splitting.
"""

import random
from collections import defaultdict

from scripts.build_prm800k_forks import split_forks_by_problem
from scripts.build_prm800k_prestudy import build_candidates
from scripts.progress_usefulness.pu_build_pairs import (
    build_split_artifacts,
    progress_forks,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _raw_sample(pid: str, sid: str) -> dict:
    """One PRM800K-shaped record with a progress(+1), neutral(0), wrong(-1) trio."""
    return {
        "problem_id": pid,
        "solution_id": sid,
        "question": {"problem": f"prob-{pid}", "ground_truth_answer": "42"},
        "label": {"steps": [
            {
                "completions": [
                    {"text": "compute 2+2=4", "rating": 1},
                    {"text": "we restate the goal", "rating": 0},
                    {"text": "so the answer is 7", "rating": -1},
                ],
                "human_completion": {"text": "gold step zero"},
            },
        ]},
    }


def _ex(pid, sid, sidx, cidx, rating):
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
    }


def _fork_map():
    # A: 1 progress, 2 neutral ; B: 2 progress, 1 neutral ; C: progress only (degenerate)
    # D: neutral+wrong only, no progress (degenerate)
    return {
        "pA::sA::0": [_ex("pA", "sA", 0, 0, 1), _ex("pA", "sA", 0, 1, 0), _ex("pA", "sA", 0, 2, 0)],
        "pB::sB::1": [_ex("pB", "sB", 1, 0, 1), _ex("pB", "sB", 1, 1, 1), _ex("pB", "sB", 1, 2, 0)],
        "pC::sC::0": [_ex("pC", "sC", 0, 0, 1)],
        "pD::sD::0": [_ex("pD", "sD", 0, 0, 0), _ex("pD", "sD", 0, 1, -1)],
    }


# --------------------------------------------------------------------------- #
# build_candidates keep_neutral behavior
# --------------------------------------------------------------------------- #

def test_build_candidates_default_drops_neutral():
    counters = defaultdict(int)
    exs, _ = build_candidates([_raw_sample("p0", "s0")], counters)
    ratings = sorted(e["rating"] for e in exs)
    assert ratings == [-1, 1]  # rating 0 dropped
    assert counters["candidate_rating_0"] == 1  # still counted


def test_build_candidates_keep_neutral_retains_zero():
    counters = defaultdict(int)
    exs, fork_map = build_candidates([_raw_sample("p0", "s0")], counters, keep_neutral=True)
    ratings = sorted(e["rating"] for e in exs)
    assert ratings == [-1, 0, 1]
    # the neutral is placed in the same fork key as its siblings (shared prefix)
    (key,) = list(fork_map.keys())
    fork_ratings = sorted(e["rating"] for e in fork_map[key])
    assert fork_ratings == [-1, 0, 1]


def test_build_candidates_neutral_label_is_not_incorrect():
    counters = defaultdict(int)
    exs, _ = build_candidates([_raw_sample("p0", "s0")], counters, keep_neutral=True)
    neutral = [e for e in exs if e["rating"] == 0]
    assert len(neutral) == 1
    assert neutral[0]["label"] == 0  # "not incorrect"; pairing keys off rating


# --------------------------------------------------------------------------- #
# progress_forks
# --------------------------------------------------------------------------- #

def test_progress_forks_keeps_only_dual_forks():
    forks = progress_forks(_fork_map())
    assert set(forks.keys()) == {"pA::sA::0", "pB::sB::1"}
    assert len(forks["pA::sA::0"]["neu"]) == 2
    assert len(forks["pB::sB::1"]["pos"]) == 2


def test_progress_forks_ignores_wrong_only_candidates():
    # rating -1 candidates must not count as neutral (fork D has 0 progress).
    forks = progress_forks(_fork_map())
    assert "pD::sD::0" not in forks


# --------------------------------------------------------------------------- #
# build_split_artifacts
# --------------------------------------------------------------------------- #

def test_pairs_share_prefix_and_map_progress_neutral():
    forks = progress_forks(_fork_map())
    items, pairs, stats = build_split_artifacts(
        "train", ["pA::sA::0"], forks, pair_mode="all", rng=random.Random(0)
    )
    by_uid = {i["item_uid"]: i for i in items}
    assert len(pairs) == 2  # 1 progress x 2 neutral
    for pr in pairs:
        prog = by_uid[pr["progress_uid"]]
        neu = by_uid[pr["neutral_uid"]]
        # shared-prefix invariant: identical fork identity for both members
        assert prog["prefix"] == neu["prefix"]
        assert (prog["problem_id"], prog["solution_id"], prog["step_idx"]) == \
               (neu["problem_id"], neu["solution_id"], neu["step_idx"])
        # progress_label mapping
        assert prog["progress_label"] == 1 and prog["rating"] == 1
        assert neu["progress_label"] == 0 and neu["rating"] == 0
    assert stats["train_pairs_if_all"] == 2
    assert stats["train_pairs_if_one"] == 1


def test_one_mode_single_pair_per_fork_and_anchor_sentinel():
    forks = progress_forks(_fork_map())
    items, pairs, _ = build_split_artifacts(
        "train", ["pA::sA::0", "pB::sB::1"], forks, pair_mode="one", rng=random.Random(1)
    )
    assert len(pairs) == 2  # one per fork
    anchors = [i for i in items if i["role"] == "anchor"]
    assert len(anchors) == 2
    assert all(a["progress_label"] == -1 and a["candidate_step"] == "" for a in anchors)


def test_item_uids_unique_and_pairs_resolve():
    forks = progress_forks(_fork_map())
    items, pairs, _ = build_split_artifacts(
        "train", ["pA::sA::0", "pB::sB::1"], forks, pair_mode="all", rng=random.Random(2)
    )
    uids = [i["item_uid"] for i in items]
    assert len(uids) == len(set(uids))
    known = set(uids)
    for pr in pairs:
        assert pr["progress_uid"] in known
        assert pr["neutral_uid"] in known
        assert pr["anchor_uid"] in known


def test_determinism_same_seed():
    forks = progress_forks(_fork_map())
    a = build_split_artifacts("train", ["pA::sA::0", "pB::sB::1"], forks, "one", random.Random(7))
    b = build_split_artifacts("train", ["pA::sA::0", "pB::sB::1"], forks, "one", random.Random(7))
    assert a[1] == b[1]  # identical pairs


# --------------------------------------------------------------------------- #
# problem-disjoint split
# --------------------------------------------------------------------------- #

def test_split_is_problem_disjoint():
    forks = progress_forks(_fork_map())
    train, val = split_forks_by_problem(forks, n_train=1, n_val=1, rng=random.Random(0))
    train_pids = {k.split("::", 1)[0] for k in train}
    val_pids = {k.split("::", 1)[0] for k in val}
    assert train_pids.isdisjoint(val_pids)
