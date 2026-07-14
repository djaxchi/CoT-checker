"""Symbolic operation labels for reasoning steps (transition_operator_v0 Stage 1).

The plan wants HIGH-PRECISION operation labels on the "parseable subset" to test
whether operation type is linearly decodable from a transition representation, and
whether same-operation steps retrieve across problems. Keyword tags
(src.analysis.contrib_cluster.tag_step) are recall-oriented and noisy; this module
adds a precision-oriented arithmetic parser that only fires when a step contains an
`a <op> b = c` identity whose top-level operation is unambiguous, and marks the
subset where the identity is NUMERICALLY VERIFIED (all-number operands, arithmetic
checks out). Steps with no parseable identity return op=None and fall back to tags.

Categories: ADD, SUB, MUL, DIV, POW. Deliberately coarse and precision-first: a
label is emitted only when one top-level operator dominates a single equation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---- latex / notation normalization ---------------------------------------

_FRAC = re.compile(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def latex_normalize(text: str) -> str:
    """Flatten the latex an arithmetic identity is likely wrapped in, into plain
    infix with * / ^ operators. Best-effort; anything unrecognized is left as-is."""
    t = text
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    t = t.replace("\\times", "*").replace("\\cdot", "*").replace("\\ast", "*")
    t = t.replace("\\div", "/")
    t = t.replace("\\%", "%")
    # \frac{a}{b} -> (a)/(b), a couple of passes for light nesting
    for _ in range(3):
        new = _FRAC.sub(r"(\1)/(\2)", t)
        if new == t:
            break
        t = new
    t = t.replace("{", "(").replace("}", ")")
    t = t.replace("\\", " ")
    t = t.replace("$", " ")
    return t


# ---- top-level operator detection -----------------------------------------

_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
_OPS = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV", "^": "POW"}
# precedence: additive dominates (last applied), then multiplicative, then power
_PRECEDENCE = [("+", "-"), ("*", "/"), ("^",)]


def _split_top_level(expr: str, ops: tuple[str, ...]) -> list[str] | None:
    """Split expr on the given operators at paren depth 0. Returns the operand list
    if exactly one such operator (possibly repeated) partitions it into >= 2
    non-empty operands, else None. A leading unary +/- is not a split point."""
    depth = 0
    parts, cur, hit = [], [], False
    for i, ch in enumerate(expr):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        # a binary operator splits only when the previous non-space char ends an
        # operand (alnum, ), ]); this rejects unary +/- and operator runs like "**"
        prev = expr[:i].rstrip()
        if depth == 0 and ch in ops and prev and (prev[-1].isalnum() or prev[-1] in ")]"):
            parts.append("".join(cur).strip())
            cur = []
            hit = True
            continue
        cur.append(ch)
    parts.append("".join(cur).strip())
    if not hit or any(p == "" for p in parts):
        return None
    return parts


def _unwrap(expr: str) -> str:
    """Strip a single fully-enclosing balanced paren pair, repeatedly."""
    expr = expr.strip()
    while len(expr) >= 2 and expr[0] == "(" and expr[-1] == ")":
        depth = 0
        for i, ch in enumerate(expr):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(expr) - 1:
                return expr  # first "(" closes before the end: not fully enclosing
        expr = expr[1:-1].strip()
    return expr


def _primary_op(lhs: str) -> tuple[str | None, list[str]]:
    """The dominant (lowest-precedence, last-applied) top-level operator in lhs."""
    expr = _unwrap(lhs)
    for group in _PRECEDENCE:
        parts = _split_top_level(expr, group)
        if parts is not None:
            # which operator of the group actually appears at depth 0
            for op in group:
                if _split_top_level(expr, (op,)) is not None:
                    return _OPS[op], parts
    return None, []


@dataclass
class OpLabel:
    op: str | None            # ADD / SUB / MUL / DIV / POW / None
    verified: bool = False    # numeric identity checked and holds


# a pure-numeric arithmetic LHS with >= 1 binary operator, then `=`, then numeric RHS
_NUM_IDENTITY = re.compile(
    r"(\(?-?[\d.]+\)?(?:\s*[-+*/^]\s*\(?-?[\d.]+\)?)+)\s*=\s*\(?(-?[\d.]+)\)?")
# a math-ish LHS (letters/digits/^ tokens with a binary operator) then `=`
_SYM_IDENTITY = re.compile(
    r"([A-Za-z0-9.^()]+(?:\s*[-+*/^]\s*[A-Za-z0-9.^()]+)+)\s*=")


def _verify(lhs: str, rhs: str) -> bool:
    try:
        return abs(eval(lhs.replace("^", "**"))       # noqa: S307
                   - eval(rhs.replace("^", "**"))) < 1e-6
    except (SyntaxError, ZeroDivisionError, ValueError, TypeError,
            OverflowError, NameError):
        return False


def symbolic_operation(text: str) -> OpLabel:
    """Precision-first op label. Prefers a numerically-verifiable identity
    `<numbers op numbers> = <number>`; otherwise a symbolic identity with an
    unambiguous top-level operator (unverified). op=None when neither is found."""
    norm = latex_normalize(text)
    m = _NUM_IDENTITY.search(norm)
    if m:
        op, _ = _primary_op(m.group(1))
        if op is not None:
            return OpLabel(op=op, verified=_verify(m.group(1), m.group(2)))
    m = _SYM_IDENTITY.search(norm)
    if m:
        op, _ = _primary_op(m.group(1))
        if op is not None:
            return OpLabel(op=op, verified=False)
    return OpLabel(op=None, verified=False)
