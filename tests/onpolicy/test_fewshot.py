"""Tests for few-shot prompting and delimiter truncation.

The risks here are not arithmetic. They are that the prompt drifts from what the
encoder rebuilds, that the exemplars stop teaching the blank-line step
convention every per-step score depends on, or that truncation eats a real
solution.
"""

from __future__ import annotations

import pytest

from scripts.generate_onpolicy_steps import split_into_steps
from src.onpolicy.fewshot import (DELIMITER, EXEMPLARS, STOP_STRING,
                                  fewshot_prompt, truncate_at_delimiter)


def test_prompt_ends_ready_for_the_model_to_write():
    p = fewshot_prompt("What is 2+2?", "gsm8k")
    assert p.endswith("Solution:\n")
    assert p.count(f"{DELIMITER}\n") == 5          # four exemplars plus the target


def test_prompt_ends_at_a_token_boundary():
    """src/onpolicy/spans.py needs this or every step span shifts by one."""
    assert fewshot_prompt("x", "math").endswith("\n")


def test_every_exemplar_teaches_the_blank_line_step_convention():
    for name, shots in EXEMPLARS.items():
        for q, a in shots:
            steps = split_into_steps(a)
            assert len(steps) >= 3, f"{name}: {q[:40]} splits into {len(steps)} steps"


def test_every_exemplar_ends_in_a_boxed_answer():
    for name, shots in EXEMPLARS.items():
        for q, a in shots:
            assert a.rstrip().endswith("}.") and "\\boxed{" in a, f"{name}: {q[:40]}"


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError):
        fewshot_prompt("x", "aime")


def test_asking_for_more_shots_than_exist_is_rejected():
    with pytest.raises(ValueError):
        fewshot_prompt("x", "gsm8k", n_shot=9)


def test_n_shot_is_a_prefix_so_the_smoke_matches_the_real_run():
    short, full = fewshot_prompt("x", "gsm8k", 2), fewshot_prompt("x", "gsm8k", 4)
    assert full.endswith(short[short.index("Problem:\nx"):])


def test_truncation_cuts_a_hallucinated_next_problem():
    text = "a step.\n\nThe answer is \\boxed{4}." + STOP_STRING + "\nnext"
    assert truncate_at_delimiter(text) == "a step.\n\nThe answer is \\boxed{4}."


def test_truncation_leaves_a_clean_completion_alone():
    text = "a step.\n\nanother step.\n\nThe answer is \\boxed{7}."
    assert truncate_at_delimiter(text) == text


def test_truncation_does_not_eat_a_solution_that_says_problem_mid_sentence():
    """'the problem' in prose must not look like a delimiter."""
    text = "Restating the problem, we want x.\n\nThe answer is \\boxed{1}."
    assert truncate_at_delimiter(text) == text


def test_truncation_never_returns_empty_for_a_leading_delimiter():
    """A completion that opens with the delimiter is degenerate, not empty.

    Returning "" would make the trace ungradeable and silently drop it; the
    caller needs to see the degenerate text and record it.
    """
    assert truncate_at_delimiter(f"{DELIMITER}\nsomething") != ""
