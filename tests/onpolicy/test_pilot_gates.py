"""The gate that matters most is the one on reranking headroom.

If any-of-N is right about as often as one sample is, every verifier gets the
same best-of-N accuracy however good it is, and the T2 correlation is measuring
sampling noise. That has to fail loudly before 20,000 trajectories are generated,
not after.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.onpolicy.pilot_gates import (  # noqa: E402
    bon_headroom, majority_answer, step_lengths,
)


def traj(pid, correct, pred, gold="4", gradeable=True):
    return {"fork_id": pid, "correct": correct, "pred": pred, "gold": gold,
            "gradeable": gradeable, "solution": "a\n\nb"}


def test_no_headroom_when_every_sample_agrees():
    """Ten identical right answers: nothing for a reranker to pick between."""
    trajs = [traj("p1", True, "4") for _ in range(10)]
    h = bon_headroom(trajs)
    assert h["oracle_at_n"] == 1.0 and h["pass_at_1"] == 1.0
    assert h["headroom_over_sample"] == 0.0
    assert h["headroom_over_sc"] == 0.0


def test_headroom_exists_when_the_right_answer_is_a_minority():
    """One right of ten, and the majority is wrong: the case a verifier can win."""
    trajs = [traj("p1", True, "4")] + [traj("p1", False, "5") for _ in range(9)]
    h = bon_headroom(trajs)
    assert h["oracle_at_n"] == 1.0
    assert h["pass_at_1"] == 0.1
    assert h["self_consistency"] == 0.0
    assert h["headroom_over_sample"] == 0.9
    assert h["headroom_over_sc"] == 1.0


def test_self_consistency_takes_the_majority_not_the_first():
    trajs = [traj("p1", False, "5"), traj("p1", True, "4"), traj("p1", True, "4")]
    assert bon_headroom(trajs)["self_consistency"] == 1.0


def test_ungradeable_trajectories_are_excluded():
    trajs = [traj("p1", True, "4"), traj("p1", False, None, gradeable=False)]
    h = bon_headroom(trajs)
    assert h["samples_per_problem"] == 1.0


def test_majority_answer_normalises_before_counting():
    assert majority_answer(["\\frac{1}{2}", "\\frac12", "3"]) is not None
    assert majority_answer([None, None]) is None


def test_step_lengths_fall_back_to_words_without_a_tokenizer():
    lens = step_lengths(["one two three", "four"], None)
    assert list(lens) == [3.0, 1.0]
