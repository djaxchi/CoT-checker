"""ReProbe's label semantics, as recovered from arXiv:2511.06209.

The convention that would have been easiest to get wrong silently is
propagation. Most of the PRM literature marks every step after the first error
as negative; the paper asks the judge to "identify and report those specific
steps" and never says the rest become negative. Propagating would roughly double
the negative class and change what the probe learns, so it is tested explicitly
rather than left to a code reading.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.onpolicy.judge_steps import (  # noqa: E402
    build_prompt_reprobe, parse_step_set, render_trace_reprobe,
    step_labels_from_faulty,
)


def test_the_gold_answer_is_shown_because_the_paper_shows_it():
    """The judge grades against a known answer: "the question, the target LLM's
    CoT steps and final answer, and the ground-truth answer"."""
    p = build_prompt_reprobe("What is 2+2?", ["a", "b"], "4")
    assert "Correct final answer: 4" in p
    assert "Step 1: a" in p and "Step 2: b" in p


def test_relevance_is_part_of_the_criterion_not_just_correctness():
    p = build_prompt_reprobe("p?", ["a"], "4")
    assert "logically correct and relevant" in p
    assert "unnecessary or redundant" in p


def test_the_faulty_set_is_parsed_as_zero_based_indices():
    assert parse_step_set("Faulty: 2, 4", 5) == [1, 3]
    assert parse_step_set("Faulty: 1", 3) == [0]


def test_none_is_a_real_answer_and_not_a_parse_failure():
    for reply in ("Faulty: NONE", "faulty: none", "Faulty: No errors"):
        assert parse_step_set(reply, 4) == [], reply


def test_an_unreadable_reply_is_a_parse_failure_not_an_empty_set():
    """Confusing "I could not tell" with "every step is sound" would silently
    turn every failure into a fully correct trajectory."""
    assert parse_step_set("I am not sure.", 4) is None
    assert parse_step_set("", 4) is None
    assert parse_step_set("Faulty:", 4) is None


def test_the_last_faulty_line_wins_when_the_judge_reasons_first():
    reply = ("Step 2 might be faulty, let me check.\n"
             "Actually step 2 is fine.\n"
             "Faulty: 3")
    assert parse_step_set(reply, 4) == [2]


def test_out_of_range_numbers_are_dropped_not_clamped():
    """A judge naming a step that does not exist has lost track of the solution;
    clamping would invent a label for a step it never looked at."""
    assert parse_step_set("Faulty: 2, 9", 4) == [1]
    assert parse_step_set("Faulty: 9", 4) is None


def test_labels_are_not_propagated_after_the_first_error():
    """The paper reports specific faulty steps and never says the rest become
    negative. Propagating is the common convention and is deliberately not used."""
    assert step_labels_from_faulty([1], 5) == [1, 0, 1, 1, 1]
    assert step_labels_from_faulty([1, 3], 5) == [1, 0, 1, 0, 1]
    assert step_labels_from_faulty([], 3) == [1, 1, 1]


def test_a_fully_correct_solution_is_all_ones():
    assert step_labels_from_faulty([], 4) == [1, 1, 1, 1]


def test_steps_are_numbered_from_one_in_the_prompt_and_zero_in_the_labels():
    text = render_trace_reprobe("p?", ["x", "y", "z"], "7")
    assert "Step 1: x" in text and "Step 3: z" in text
    assert step_labels_from_faulty(parse_step_set("Faulty: 3", 3), 3) == [1, 1, 0]
