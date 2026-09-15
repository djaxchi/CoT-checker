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


# ---- prompt-style dispatch ------------------------------------------------
#
# Two consumers reconstruct the sampler's context to recover generative states:
# the logprob encoder and the online rejection loop. Both used to hardcode the
# zero-shot prompt. That failure is silent, since the forward pass still runs
# and the numbers still look plausible while describing a context the model
# never saw, so it gets its own tests.

def test_context_zero_style_is_byte_identical_to_the_old_path():
    from src.onpolicy.prompts import context, generation_prefix
    assert context("zero", "P", "s1") == generation_prefix("P", "s1")
    assert context("zero", "P") == generation_prefix("P", "")


def test_context_fewshot_starts_from_the_fewshot_prompt():
    from src.onpolicy.prompts import context
    got = context("fewshot", "P", "", "gsm8k")
    assert got == fewshot_prompt("P", "gsm8k")


def test_context_fewshot_appends_prior_steps_with_the_step_separator():
    from src.onpolicy.prompts import context
    got = context("fewshot", "P", "step one\n\nstep two", "gsm8k")
    assert got == fewshot_prompt("P", "gsm8k") + "step one\n\nstep two\n\n"


def test_context_from_row_defaults_to_zero_for_legacy_rows():
    """Rows written before tts_roster_v1 carry no style and must not change."""
    from src.onpolicy.prompts import context_from_row, generation_prefix
    assert context_from_row({"problem": "P"}) == generation_prefix("P", "")


def test_context_from_row_uses_the_rows_own_dataset():
    from src.onpolicy.prompts import context_from_row
    row = {"problem": "P", "prompt_style": "fewshot", "dataset": "math", "n_shot": 4}
    assert context_from_row(row) == fewshot_prompt("P", "math")
    assert context_from_row(row) != fewshot_prompt("P", "gsm8k")


def test_context_rejects_an_unknown_style_rather_than_defaulting():
    from src.onpolicy.prompts import context
    with pytest.raises(ValueError):
        context("chatml", "P")


# ---- answer truncation ----------------------------------------------------
#
# The sampler must cut a trace at the answer it states, because math_grade takes
# the LAST boxed answer. Measured on a 240-trace smoke: 16 traces were graded on
# a hallucinated second problem, and 4 stated a wrong answer, restarted,
# re-solved and were credited for the do-over. The second class is why this
# belongs in the sampler: a trace running its own informal best-of-2 contaminates
# the comparison this study exists to make.

def test_answer_cut_keeps_the_stated_answer_and_drops_the_restart():
    from src.onpolicy.fewshot import truncate_at_answer
    text = ("x = 4.\n\nThe answer is \\boxed{4}.\n\n"
            "The given equation is different.\n\nSo x = 2.\n\nThe answer is \\boxed{2}.")
    assert truncate_at_answer(text) == "x = 4.\n\nThe answer is \\boxed{4}."


def test_answer_cut_keeps_trailing_punctuation_on_the_line():
    """Cutting at the closing brace would leave a sentence fragment for the
    step splitter."""
    from src.onpolicy.fewshot import truncate_at_answer
    assert truncate_at_answer("The answer is $\\boxed{7}$.").endswith("$.")


def test_answer_cut_is_inert_without_the_taught_phrase():
    from src.onpolicy.fewshot import truncate_at_answer
    text = "We compute \\boxed{5} as an intermediate.\n\nThen more work."
    assert truncate_at_answer(text) == text


def test_answer_cut_does_not_fire_on_a_bare_intermediate_box():
    """Anchoring on any \\boxed{} instead of the phrase ate real solutions that
    box an intermediate first; it cost one extra bad flip when measured."""
    from src.onpolicy.fewshot import truncate_at_answer
    text = ("Area is \\boxed{12} so far.\n\nContinuing.\n\n"
            "The answer is \\boxed{24}.\n\nJunk after.")
    assert truncate_at_answer(text) == ("Area is \\boxed{12} so far.\n\nContinuing.\n\n"
                                        "The answer is \\boxed{24}.")


def test_answer_cut_leaves_an_unfinished_trace_alone():
    from src.onpolicy.fewshot import truncate_at_answer
    text = "step one.\n\nstep two.\n\nstill working"
    assert truncate_at_answer(text) == text


def test_answer_cut_handles_no_trailing_newline():
    from src.onpolicy.fewshot import truncate_at_answer
    text = "The answer is \\boxed{3}."
    assert truncate_at_answer(text) == text
