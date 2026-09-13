"""Mapping a generated solution's steps and final answer onto token positions.

The confidence rules in src/analysis/token_confidence.py are per-token, the
verifier is per-step, and the head-to-head only means anything if both read the
same step boundaries. This module produces those boundaries from the same
blank-line convention `scripts/generate_onpolicy_steps.py` used to write the
step items, so a step span here is the same step the verifier scored.

Spans are half-open [start, end) indices into the *generated* token sequence,
with position 0 the first token after the prompt.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

_BOXED = re.compile(r"\\boxed\s*\{")


def step_token_spans(prompt: str, steps: Sequence[str],
                     n_tokens: Callable[[str], int]) -> list[tuple[int, int]]:
    """Token spans of each step, measured by incremental re-tokenisation.

    `n_tokens` returns the token count of a string under the run's tokenizer.
    Boundaries come from tokenising the prompt plus the first k steps and
    subtracting the prompt length, which is the same construction
    `src/onpolicy/prompts.py` uses to rebuild a step's context. Tokenising each
    step in isolation and concatenating would not work: a tokenizer merges
    across the join, so the spans would drift from the real sequence.

    The final span is left open-ended by the caller clipping to the true
    sequence length; steps beyond a truncated generation collapse to empty and
    are dropped downstream rather than silently reported as zero-confidence.

    One precondition the caller owns: `prompt` must end at a token boundary. If
    it ends mid-word the tokenizer merges the prompt's last token with the
    step's first and every span shifts by one. `generation_prompt` ends on a
    newline, which satisfies this, and `verify_spans_cover` below is the check
    that it still holds for any new prompt format.
    """
    base = n_tokens(prompt)
    spans: list[tuple[int, int]] = []
    prev = 0
    for k in range(len(steps)):
        upto = n_tokens(prompt + "\n\n".join(steps[:k + 1]))
        end = max(prev, upto - base)
        spans.append((prev, end))
        prev = end
    return spans


def answer_char_span(solution: str) -> tuple[int, int] | None:
    """Character span of the contents of the last \\boxed{...} in the solution.

    The last one, not the first: models restate. Brace matching rather than a
    lazy regex because answers contain braces, as in \\boxed{\\frac{1}{2}}.
    """
    last = None
    for m in _BOXED.finditer(solution):
        last = m
    if last is None:
        return None
    i = last.end()
    depth = 1
    while i < len(solution) and depth:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return (last.end(), i)
        i += 1
    return None


def char_span_to_token_span(offsets: Sequence[tuple[int, int]],
                            char_span: tuple[int, int]) -> tuple[int, int]:
    """Token span covering a character span, from a fast tokenizer's offsets.

    Any token that overlaps the character range is included, so a token
    straddling the opening brace is not dropped.
    """
    a, b = char_span
    idx = [i for i, (s, e) in enumerate(offsets) if e > a and s < b]
    return (idx[0], idx[-1] + 1) if idx else (0, 0)


def verify_spans_cover(spans: Sequence[tuple[int, int]], n_generated: int,
                       tol: int = 2) -> bool:
    """Whether the spans tile the generated sequence, within `tol` tokens.

    The failure this catches is a prompt that does not end at a token boundary,
    which shifts every span by one and would otherwise produce plausible but
    systematically misaligned per-step confidences. Run it on a sample of real
    trajectories before trusting any number built on these spans.
    """
    if not spans:
        return n_generated == 0
    if spans[0][0] != 0:
        return False
    for (_, e1), (s2, _) in zip(spans, spans[1:]):
        if e1 != s2:
            return False
    return abs(spans[-1][1] - n_generated) <= tol
