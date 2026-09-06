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


def test_the_verdict_is_read_from_the_harmony_final_channel():
    """GPT-OSS thinks aloud in an analysis channel first. Parsing the whole reply
    would read a hypothesis from the thinking as the verdict."""
    from scripts.onpolicy.judge_steps import harmony_final
    reply = ("analysisLet me check. Step 2 might be faulty: 7+5=12 is right. "
             "Faulty: 2 would be wrong here.assistantfinalFaulty: 4")
    assert harmony_final(reply).strip() == "Faulty: 4"
    assert parse_step_set(reply, 5) == [3]


def test_a_plain_reply_with_no_channel_marker_is_parsed_whole():
    from scripts.onpolicy.judge_steps import harmony_final
    assert harmony_final("Faulty: 2") == "Faulty: 2"
    assert parse_step_set("Faulty: 2", 3) == [1]


def test_an_analysis_that_never_reaches_the_final_channel_fails_to_parse():
    """This is the 81% case from the smoke run: the model ran out of budget
    mid-thought. It must be a parse failure, not a label."""
    from scripts.onpolicy.judge_steps import harmony_final
    truncated = "analysisWe need to identify faulty steps. Let's examine. Step 1"
    assert harmony_final(truncated) is None
    assert parse_step_set(truncated, 5) is None
    # the danger is concrete: the analysis is full of sentences that read like a
    # verdict, and parsing it would have produced a label from the thinking
    assert parse_step_set("analysisStep 2 might be faulty, checking...", 5) is None


def test_an_oom_splits_the_batch_instead_of_killing_the_run():
    """Job 443140 died eight minutes into a six-hour shard because one batch of
    unusually long traces did not fit. Splitting on demand costs nothing on the
    batches that fit and saves the run on the ones that do not."""
    import sys
    import types
    from scripts.onpolicy import judge_local_reprobe as jl

    calls = []

    class FakeOOM(Exception):
        pass

    class FakeTorch(types.SimpleNamespace):
        OutOfMemoryError = FakeOOM

    def fake_generate(model, tok, prompts, device, args):
        return jl.generate_with_oom_retry(model, tok, prompts, device, args)

    class Tok:
        pad_token_id = 0
        def __call__(self, prompts, **kw):
            class E(dict):
                def to(self, d): return self
            return E(input_ids=_Arr(len(prompts)))
        def batch_decode(self, x, **kw): return ["ok"] * x.n

    class _Arr:
        def __init__(self, n): self.n = n
        @property
        def shape(self): return (self.n, 3)
        def __getitem__(self, k): return _Arr(self.n)

    class Model:
        def generate(self, **kw):
            n = kw["input_ids"].n
            calls.append(n)
            if n > 2:
                raise FakeOOM("out of memory")
            return _Arr(n)

    real_torch = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch(
        OutOfMemoryError=FakeOOM,
        cuda=types.SimpleNamespace(empty_cache=lambda: None),
        no_grad=lambda: __import__("contextlib").nullcontext())
    try:
        class A:
            max_new_tokens = 8
        out = jl.generate_with_oom_retry(Model(), Tok(), ["p"] * 8, None, A())
    finally:
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            del sys.modules["torch"]
    assert len(out) == 8                    # nothing was dropped
    assert 8 in calls and min(calls) <= 2   # it tried big, then split down
