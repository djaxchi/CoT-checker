"""Agreement between two labellers that mean different things."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.onpolicy_label_agreement import agreement, load  # noqa: E402


def rows(pairs, n_steps=5):
    return {uid: {"traj_uid": uid, "first_error": fe, "n_steps": n_steps}
            for uid, fe in pairs}


def test_identical_labels_agree_exactly():
    a = rows([("t0", 1), ("t1", 3), ("t2", -1)])
    r = agreement(a, dict(a))
    assert r["exact"] == 1.0
    assert r["neither"] == 1 / 3


def test_off_by_one_is_reported_separately_from_exact():
    a = rows([("t0", 1), ("t1", 2)])
    b = rows([("t0", 2), ("t1", 2)])
    r = agreement(a, b)
    assert r["exact"] == 0.5
    assert r["within_one"] == 1.0


def test_one_finding_an_error_the_other_misses_is_counted_on_its_own_side():
    a = rows([("t0", 2), ("t1", -1)])
    b = rows([("t0", -1), ("t1", -1)])
    r = agreement(a, b)
    assert r["only_a_found_one"] == 0.5
    assert r["only_b_found_one"] == 0.0
    assert r["neither"] == 0.5


def test_relative_position_is_normalised_by_trace_length():
    a = rows([("t0", 4)], n_steps=5)      # last step of five -> 1.0
    b = rows([("t0", 0)], n_steps=5)      # first step        -> 0.0
    r = agreement(a, b)
    assert r["mean_relative_position_a"] == 1.0
    assert r["mean_relative_position_b"] == 0.0


def test_only_shared_trajectories_are_compared(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps({"traj_uid": f"t{i}", "first_error": 1,
                                       "n_steps": 3}) for i in range(5)) + "\n")
    b.write_text("\n".join(json.dumps({"id": f"t{i}", "first_error": 1,
                                       "n_steps": 3}) for i in range(3)) + "\n")
    r = agreement(load(a), load(b))
    assert r["n_shared"] == 3
