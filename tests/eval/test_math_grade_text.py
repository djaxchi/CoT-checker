"""The \\text{} wrapper is a unit annotation only when something precedes it.

MATH gold answers are sometimes words: "\\text{Evelyn}", "\\text{even}". The
original rule stripped everything from "\\text{" onward, so every such answer
normalised to the empty string, compared equal to every other one, and marked
"\\text{Carla}" correct against a gold of "\\text{Evelyn}".
"""

from __future__ import annotations

from src.eval.math_grade import grade, normalize_answer


def test_a_word_answer_survives_normalisation():
    assert normalize_answer(r"\text{Evelyn}") == "Evelyn"
    assert normalize_answer(r"\text{even}") == "even"


def test_two_different_word_answers_stay_different():
    """The bug: both became "" and every text answer matched every other."""
    assert normalize_answer(r"\text{Carla}") != normalize_answer(r"\text{Evelyn}")


def test_a_word_answer_does_not_match_a_blank_prediction():
    assert normalize_answer(r"\text{Evelyn}") != normalize_answer("")


def test_wrapped_and_bare_forms_agree():
    """A model that writes Evelyn without the wrapper must still be right."""
    assert normalize_answer("Evelyn") == normalize_answer(r"\text{Evelyn}")


def test_a_trailing_unit_is_still_stripped():
    assert normalize_answer(r"5 \text{ cm}") == "5"
    assert normalize_answer(r"12\text{ meters}") == "12"


def test_a_wrapper_after_content_is_treated_as_a_unit():
    """Once the leading wrapper is unwrapped, what follows is a unit again.

    "\\text{Evelyn}\\text{x}" unwraps to "Evelyn\\text{x}", which is content
    plus a trailing annotation, so the annotation is stripped. That is the same
    rule applied twice, not a special case.
    """
    assert normalize_answer(r"\text{Evelyn}\text{x}") == "Evelyn"


def test_an_unbalanced_wrapper_is_left_alone_rather_than_emptied():
    assert normalize_answer(r"\text{") == r"\text{"


def test_grading_rejects_the_wrong_word():
    g = grade(r"work\n\nThe answer is \boxed{\text{Carla}}.", r"\text{Evelyn}")
    assert g["correct"] is False


def test_grading_accepts_the_right_word():
    g = grade(r"work\n\nThe answer is \boxed{\text{Evelyn}}.", r"\text{Evelyn}")
    assert g["correct"] is True
