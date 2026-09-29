"""The panel decides with one checker and records every checker's score."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.onpolicy.online_checkers import PanelChecker  # noqa: E402


class _Cell:
    def __init__(self, name):
        self.name, self.seconds = name, 0.0


class _Gen:
    seconds_backbone = 0.0

    def __init__(self):
        self.cells = [_Cell("a"), _Cell("b")]

    def score_all(self, problem, prior, cand, dataset="", n_shot=4):
        return {"a": 0.1 * len(cand), "b": 0.9}


class _PRM:
    name, seconds = "prm_qwen25_math_7b", 0.0

    def score(self, problem, prior, cand):
        return 0.5


def test_active_member_decides_and_all_are_recorded():
    p = PanelChecker(_Gen(), _PRM(), "a")
    assert p.score_steps("q", [], ["xy", "xyz"]) == pytest.approx([0.2, 0.3])
    assert p.last[1] == pytest.approx({"a": 0.3, "b": 0.9, "prm_qwen25_math_7b": 0.5})
    assert p.calls == 2
    assert PanelChecker(_Gen(), _PRM(), "prm_qwen25_math_7b").score_steps("q", [], ["x"]) == [0.5]


def test_unknown_active_member_is_refused():
    with pytest.raises(ValueError):
        PanelChecker(_Gen(), None, "prm_qwen25_math_7b")


def test_none_active_scores_zero_so_plain_and_blind_never_reject_on_it():
    p = PanelChecker(_Gen(), None, "none")
    assert p.score_steps("q", [], ["x"]) == [0.0]
    assert p.last[0]["a"] == pytest.approx(0.1)
