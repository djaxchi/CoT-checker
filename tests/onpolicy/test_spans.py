"""Tests for step and answer span mapping."""

from __future__ import annotations

from src.onpolicy.spans import (answer_char_span, char_span_to_token_span,
                                step_token_spans)


def _words(s: str) -> int:
    """A tokenizer stub: one token per whitespace-separated word."""
    return len(s.split())


def test_step_spans_are_contiguous_and_cover_the_generation():
    prompt, steps = "P P P ", ["a b", "c", "d e f"]
    spans = step_token_spans(prompt, steps, _words)
    assert spans == [(0, 2), (2, 3), (3, 6)]
    assert spans[0][0] == 0
    for (_, e1), (s2, _) in zip(spans, spans[1:]):
        assert e1 == s2


def test_step_spans_never_go_backwards_on_a_merging_tokenizer():
    """A tokenizer can merge across the join and shorten the total.

    The span must clamp to empty rather than produce end < start, which would
    slice backwards and silently return the wrong tokens.
    """
    def shrinking(s: str) -> int:
        return max(1, len(s.split()) - s.count("x"))
    spans = step_token_spans("P", ["x x x", "y"], shrinking)
    assert all(e >= s for s, e in spans)


def test_answer_span_takes_the_last_boxed():
    sol = r"first \boxed{1} then really \boxed{42}"
    a, b = answer_char_span(sol)
    assert sol[a:b] == "42"


def test_answer_span_matches_nested_braces():
    sol = r"so \boxed{\frac{1}{2}}"
    a, b = answer_char_span(sol)
    assert sol[a:b] == r"\frac{1}{2}"


def test_answer_span_none_when_unboxed_or_unclosed():
    assert answer_char_span("no answer here") is None
    assert answer_char_span(r"\boxed{oops") is None


def test_char_span_to_token_span_includes_overlapping_tokens():
    offsets = [(0, 3), (3, 7), (7, 10), (10, 14)]
    assert char_span_to_token_span(offsets, (4, 9)) == (1, 3)
    assert char_span_to_token_span(offsets, (100, 110)) == (0, 0)


def test_verify_spans_cover_accepts_a_clean_tiling():
    from src.onpolicy.spans import verify_spans_cover
    assert verify_spans_cover([(0, 3), (3, 7)], 7)
    assert verify_spans_cover([(0, 3), (3, 7)], 8)          # within tol


def test_verify_spans_cover_catches_the_off_by_one_prompt_boundary():
    """A prompt not ending at a token boundary shifts every span by one."""
    from src.onpolicy.spans import verify_spans_cover
    assert not verify_spans_cover([(1, 4), (4, 8)], 8)      # does not start at 0
    assert not verify_spans_cover([(0, 3), (4, 8)], 8)      # gap between steps
    assert not verify_spans_cover([(0, 3), (3, 7)], 40)     # far short of the trace
