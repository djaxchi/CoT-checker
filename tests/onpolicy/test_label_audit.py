"""The label audit has to catch the pathologies that training curves hide.

Three failure modes matter and each has a test that constructs it: a judge that
marks faults at the same rate whatever the outcome (reading nothing), one that
marks faults in exactly the failing trajectories (copying the grader rather than
reading the reasoning), and one that marks most steps of a trace faulty (no
localisation signal left, and an inflated positive class).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.onpolicy_label_audit import audit, checks  # noqa: E402


def row(uid, correct, faulty, n_steps=6, parse_ok=True, pid=None):
    return {"traj_uid": uid, "id": uid, "problem_id": pid or uid[:3],
            "traj_correct": correct, "faulty_steps": faulty, "n_steps": n_steps,
            "first_error": (min(faulty) if faulty else -1), "parse_ok": parse_ok}


def named(rep):
    return {n: ok for n, ok, _ in checks(rep)}


def test_a_healthy_label_set_passes_every_check():
    rows = ([row(f"c{i}", True, [] if i % 4 else [2]) for i in range(40)] +
            [row(f"w{i}", False, [i % 5]) for i in range(40)])
    assert all(named(audit(rows)).values())


def test_a_judge_that_reads_nothing_is_caught():
    """Same fault rate whether or not the answer was right."""
    rows = ([row(f"c{i}", True, [1]) for i in range(40)] +
            [row(f"w{i}", False, [1]) for i in range(40)])
    assert named(audit(rows))["judge discriminates"] is False


def test_a_judge_copying_the_grader_is_caught():
    """Faults in exactly the failing trajectories and never elsewhere means the
    label is the final answer wearing a disguise."""
    rows = ([row(f"c{i}", True, []) for i in range(40)] +
            [row(f"w{i}", False, [2]) for i in range(40)])
    assert named(audit(rows))["not just copying the grader"] is False


def test_traces_marked_almost_entirely_faulty_are_caught():
    rows = ([row(f"c{i}", True, []) for i in range(40)] +
            [row(f"w{i}", False, [0, 1, 2, 3, 4], n_steps=6) for i in range(40)])
    n = named(audit(rows))
    assert n["few degenerate traces"] is False or n["positives are not the majority"] is False


def test_errors_all_at_step_zero_are_caught():
    rows = ([row(f"c{i}", True, [] if i % 3 else [2]) for i in range(40)] +
            [row(f"w{i}", False, [0]) for i in range(40)])
    assert named(audit(rows))["errors are not all at step 0"] is False


def test_parse_failures_and_duplicates_are_counted_not_dropped():
    rows = [row("a", True, []), row("a", True, []), row("b", False, [1], parse_ok=False)]
    rep = audit(rows)
    assert rep["duplicate_ids"] == 1
    assert rep["n_annotated"] == 3 and rep["n_parsed"] == 2
    assert named(rep)["no duplicate ids"] is False


def test_the_measured_partial_run_would_pass():
    """The real numbers from the first 96 annotations: 0.262 false alarms against
    0.863 coverage. Discrimination 0.601, comfortably clear of the floor."""
    rows = ([row(f"c{i}", True, [1] if i < 26 else []) for i in range(100)] +
            [row(f"w{i}", False, [i % 4] if i < 86 else []) for i in range(100)])
    rep = audit(rows)
    assert rep["discrimination"] > 0.5
    assert all(named(rep).values())
