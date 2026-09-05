"""Per-step labels have to survive the encoder, and their polarity has to flip.

Two conventions meet here and they disagree. ReProbe reports a SET of faulty
steps and its labels are 1 for correct; this project's stores use y=1 for an
INCORRECT step and, under the ProcessBench convention, the only positive in a
trace is the first-error step. Encoding ReProbe labels through the old path
would train on one positive per trace instead of the set, and encoding them
without flipping would train the probe to predict exactly the opposite of what
every other cell predicts, which would still produce plausible-looking curves.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from scripts.encode_processbench_token_store import flatten  # noqa: E402


def trace(**kw):
    base = {"id": "t0", "problem": "p?", "steps": ["a", "b", "c"], "label": 1}
    base.update(kw)
    return base


def y_of(row):
    """The label the encoder would write for this flattened row."""
    return (1 - int(row["step_label"])) if row.get("step_label") is not None \
        else int(row["step_idx"] == row["label"])


def test_without_step_labels_only_the_first_error_step_is_positive():
    rows = flatten([trace()], "s")
    assert [y_of(r) for r in rows] == [0, 1, 0]


def test_a_step_label_vector_marks_every_faulty_step():
    """ReProbe's set convention: steps 1 and 3 faulty, step 2 fine."""
    rows = flatten([trace(step_labels=[0, 1, 0])], "s")
    assert [y_of(r) for r in rows] == [1, 0, 1]


def test_the_polarity_is_inverted_because_the_two_conventions_disagree():
    """ReProbe: 1 means correct. This project: y=1 means incorrect."""
    rows = flatten([trace(step_labels=[1, 1, 1])], "s")
    assert [y_of(r) for r in rows] == [0, 0, 0]
    rows = flatten([trace(step_labels=[0, 0, 0])], "s")
    assert [y_of(r) for r in rows] == [1, 1, 1]


def test_a_misaligned_label_vector_is_refused_not_truncated():
    """Silently zipping a short vector against the steps would train on the
    wrong steps and nothing downstream would look wrong."""
    with pytest.raises(ValueError, match="step_labels"):
        flatten([trace(step_labels=[1, 0])], "s")
    with pytest.raises(ValueError, match="step_labels"):
        flatten([trace(step_labels=[1, 0, 1, 0])], "s")


def test_the_trace_level_fields_still_ride_along():
    rows = flatten([trace(step_labels=[1, 0, 1], traj_correct=False,
                          problem_id="p9")], "s")
    assert {r["problem_id"] for r in rows} == {"p9"}
    assert {r["traj_correct"] for r in rows} == {False}
    assert [r["global_index"] for r in rows] == [0, 1, 2]
