"""The converter must refuse any trace the frontier would silently misread."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.onpolicy.pb_scores_to_tts import convert  # noqa: E402

TRAJ = [{"traj_uid": "a", "fork_id": "p", "gradeable": True, "correct": True,
         "solution": "one.\n\ntwo."},
        {"traj_uid": "b", "fork_id": "p", "gradeable": False, "correct": False,
         "solution": "x"}]


def test_converts_and_skips_ungradeable():
    out = convert([{"id": "a", "scores": [0.1, 0.9]}], TRAJ, "cell")
    assert out == [{"traj_uid": "a", "problem_id": "p", "correct": True, "n_steps": 2,
                    "cell": "cell", "scores": [0.1, 0.9]}]


def test_missing_trace_fails():
    with pytest.raises(ValueError, match="no scores"):
        convert([], TRAJ, "cell")


def test_dropped_step_fails():
    with pytest.raises(ValueError, match="1 step scores for 2 steps"):
        convert([{"id": "a", "scores": [0.1]}], TRAJ, "cell")
