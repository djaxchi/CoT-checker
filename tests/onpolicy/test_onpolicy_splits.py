"""Splitting the on-policy pool, where a leak would be invisible.

Ten samples of one problem exist in this pool. Splitting by trajectory would put
some on each side and let the probe memorise the problem instead of learning what
a faulty step looks like, and the validation curve would look better for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.onpolicy.build_onpolicy_splits import join_labels, split_by_problem  # noqa: E402


def trace(uid, pid, n_steps=3):
    return {"id": uid, "problem_id": pid, "problem": "p?",
            "steps": ["a"] * n_steps, "gold": "4"}


def test_no_problem_appears_on_both_sides():
    traces = [trace(f"t{p}_{g}", f"p{p}") for p in range(20) for g in range(10)]
    train, val, info = split_by_problem(traces, 0.2, seed=0)
    assert not ({t["problem_id"] for t in train} & {t["problem_id"] for t in val})
    assert len(train) + len(val) == len(traces)
    assert info["n_val_problems"] == 4


def test_the_split_is_deterministic_under_the_seed():
    traces = [trace(f"t{p}", f"p{p}") for p in range(30)]
    a = split_by_problem(traces, 0.2, seed=7)[2]["val_problem_ids"]
    b = split_by_problem(traces, 0.2, seed=7)[2]["val_problem_ids"]
    c = split_by_problem(traces, 0.2, seed=8)[2]["val_problem_ids"]
    assert a == b and a != c


def test_labels_of_the_wrong_length_are_dropped_not_attached():
    """A judge that named steps of a different length was not looking at this
    trajectory; attaching the vector would train on labels for something else."""
    traces = [trace("t0", "p0", n_steps=3)]
    labels = {"t0": {"parse_ok": True, "step_labels": [1, 0]}}
    out, tally = join_labels(traces, labels, 2)
    assert out == []
    assert tally["label_length_mismatch"] == 1


def test_a_judge_parse_failure_is_not_treated_as_all_correct():
    traces = [trace("t0", "p0")]
    out, tally = join_labels(traces, {"t0": {"parse_ok": False}}, 2)
    assert out == []
    assert tally["judge_parse_failed"] == 1


def test_the_first_error_index_is_derived_from_the_faulty_set():
    traces = [trace("t0", "p0", n_steps=4)]
    out, _ = join_labels(traces, {"t0": {"parse_ok": True,
                                         "step_labels": [1, 1, 0, 0]}}, 2)
    assert out[0]["faulty_steps"] == [2, 3]
    assert out[0]["label"] == 2
    assert out[0]["step_labels"] == [1, 1, 0, 0]


def test_a_fully_correct_trace_gets_minus_one():
    traces = [trace("t0", "p0", n_steps=3)]
    out, _ = join_labels(traces, {"t0": {"parse_ok": True,
                                         "step_labels": [1, 1, 1]}}, 2)
    assert out[0]["label"] == -1 and out[0]["faulty_steps"] == []
