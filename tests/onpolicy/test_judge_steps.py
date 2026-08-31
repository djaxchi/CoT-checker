"""The judge's arithmetic, where the silent failures live.

Two of these matter more than the rest. Off-by-one: the prompt numbers steps from
1 and every label downstream is 0-based, so a convention slip would shift every
error by one step and still look entirely plausible. And the baselines: a judge
that has only learned "mistakes come near the end" scores respectably on F1_PB,
so the degenerate strategies have to be scored on the same traces or there is no
way to see it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.onpolicy.judge_steps import (  # noqa: E402
    NO_ERROR, baselines, build_prompt, certify, parse_answer, pb_metrics,
    render_trace, trace_outcome, vote,
)


def test_step_numbers_are_one_based_in_the_prompt_and_zero_based_out():
    text = render_trace("p?", ["a", "b", "c"])
    assert "Step 1: a" in text and "Step 3: c" in text
    assert parse_answer("Answer: 1", 3) == 0
    assert parse_answer("Answer: 3", 3) == 2


def test_no_error_survives_as_minus_one():
    assert parse_answer("Answer: -1", 4) == NO_ERROR


def test_a_zero_is_a_parse_failure_not_a_guess():
    """0 means the model used another convention; guessing shifts every label."""
    assert parse_answer("Answer: 0", 4) is None


def test_out_of_range_and_unreadable_replies_fail_to_parse():
    assert parse_answer("Answer: 9", 4) is None
    assert parse_answer("I am not sure.", 4) is None


def test_the_answer_line_wins_over_stray_numbers():
    assert parse_answer("Step 2 uses 7 and 12.\nAnswer: 3", 5) == 2


def test_the_judge_is_told_the_outcome_and_told_it_the_same_way_in_both_sets():
    """Certification and deployment must be the same task, or the certification
    number describes a judge we do not run. The outcome is the one piece of
    supervision both sets carry, under different field names."""
    wrong = build_prompt("p?", ["a"], outcome=False)
    right = build_prompt("p?", ["a"], outcome=True)
    assert "INCORRECT final answer" in wrong
    assert "CORRECT final answer" in right and "answer -1 only if" in right
    assert "Correct final answer:" not in wrong          # the gold answer is not shown
    assert trace_outcome({"final_answer_correct": False}) is False   # ProcessBench
    assert trace_outcome({"traj_correct": True}) is True             # on-policy
    assert trace_outcome({"id": "x"}) is None


def test_gold_answer_is_available_but_off_by_default():
    assert "Correct final answer: 4" in build_prompt("p?", ["a"], None, "4")


def test_votes_take_the_majority_and_break_ties_early():
    assert vote([2, 2, 5]) == 2
    assert vote([5, 2]) == 2
    assert vote([None, None]) is None


def test_pb_metrics_match_the_harness_definition():
    m = pb_metrics([1, -1, 0], [1, -1, 2])
    assert m["Acc_error"] == 0.5 and m["Acc_correct"] == 1.0
    assert m["F1_PB"] == 2 * 0.5 * 1.0 / 1.5
    assert m["exact_match_all"] == 2 / 3


def test_always_last_step_is_scored_alongside_the_judge():
    gold = [2, 3, NO_ERROR]
    b = baselines(gold, [3, 4, 5])
    assert b["always_last_step"]["Acc_error"] == 1.0    # errors sit at the end here
    assert b["always_last_step"]["Acc_correct"] == 0.0
    assert b["always_no_error"]["Acc_correct"] == 1.0


def test_unparseable_replies_count_as_no_error_rather_than_being_dropped():
    traces = [{"id": "t0", "problem": "p", "steps": ["a", "b"], "label": 1},
              {"id": "t1", "problem": "p", "steps": ["a", "b"], "label": -1}]
    rows = [{"id": "t0", "first_error": -1, "parse_ok": False},
            {"id": "t1", "first_error": -1, "parse_ok": True}]
    rep = certify(rows, traces)
    assert rep["judge"]["Acc_error"] == 0.0        # the failure is not forgiven
    assert rep["judge"]["Acc_correct"] == 1.0
    assert rep["parse_failure_rate"] == 0.5


def test_relative_position_exposes_a_late_biased_judge():
    traces = [{"id": f"t{i}", "problem": "p", "steps": ["a"] * 5, "label": 1}
              for i in range(4)]
    rows = [{"id": f"t{i}", "first_error": 4, "parse_ok": True} for i in range(4)]
    mp = certify(rows, traces)["mean_relative_position"]
    assert mp["predicted"] == 1.0 and mp["true"] == 0.25


def test_with_reasoning_the_last_answer_line_is_the_verdict():
    """A judge that reasons first may say "answer" on the way; the verdict is
    the line it ends on, and reading the first one would take a hypothesis for a
    conclusion."""
    reply = ("Step 1 looks fine.\n"
             "Step 2's answer of 12 does not follow from step 1.\n"
             "Answer: 2")
    assert parse_answer(reply, 4) == 1


def test_the_reasoning_prompt_asks_for_an_ordered_check_and_a_final_line():
    p = build_prompt("p?", ["a", "b"], outcome=False, cot=True)
    assert "Check the steps in order" in p
    assert p.rstrip().endswith("Check:")
    assert "Answer:" in p
