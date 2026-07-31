"""Unit tests for the PRM800K next-step generation helpers (pure text logic)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_prm800k_next_step import (  # noqa: E402
    build_gen_prompt, extract_next_step, solution_so_far,
)


def test_solution_so_far_joins_nonempty():
    assert solution_so_far("step one", "step two") == "step one\n\nstep two"
    assert solution_so_far("", "only step") == "only step"          # step 0, empty prefix
    assert solution_so_far("  a  ", "  b ") == "a\n\nb"              # trimmed


def test_build_gen_prompt_shape():
    pr = build_gen_prompt("2+2?", "first thoughts", "compute it")
    assert pr.startswith("Problem:\n2+2?\n\nSolution:\n")
    assert pr.endswith("\n\n")                                       # ready to continue
    assert "first thoughts\n\ncompute it" in pr


def test_extract_next_step_takes_first_chunk():
    assert extract_next_step("the next step.\n\nand a later one") == "the next step."
    assert extract_next_step("  spaced step \n\n junk") == "spaced step"
    assert extract_next_step("") == ""
    assert extract_next_step("single line no break") == "single line no break"
