"""MATH-style answer equivalence grading for PRM800K (Stage 1 on-policy labeling).

PRM800K answers are LaTeX (``\\frac{1}{3}``, ``50^{\\circ}``, ``83\\%``,
``4+3\\sqrt{2}``, ``-\\frac{\\sqrt{3}}{2}``), so on-policy generated solutions cannot be
graded by string match. This module implements the canonical Hendrycks MATH
``is_equiv`` string-normalization grader (handles fractions, sqrt, units, percent,
degrees) with sympy used only as an *optional* equivalence booster for cases that
survive normalization but are still equal (commuted sums, simplifiable expressions).

sympy is imported lazily and the grader degrades gracefully to pure string
normalization if it is unavailable (offline Tamia wheelhouse may lack it).
"""

from __future__ import annotations

import re

__all__ = ["last_boxed_only_string", "remove_boxed", "normalize_answer",
           "is_equiv", "extract_final_answer", "grade"]


# --------------------------------------------------------------------------- #
# \boxed extraction
# --------------------------------------------------------------------------- #

def last_boxed_only_string(string: str) -> str | None:
    """Return the last ``\\boxed{...}`` (or ``\\fbox{...}``) substring, braces matched."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    depth = 0
    right_brace_idx = None
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def remove_boxed(s: str | None) -> str | None:
    """Strip the ``\\boxed{...}`` wrapper, returning the inner content."""
    if s is None:
        return None
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    # \boxed 1.23  (no braces) form
    m = re.match(r"\\boxed\s+(.+)", s)
    if m:
        return m.group(1)
    return s


def extract_final_answer(solution: str) -> str | None:
    """Best-effort final answer from a generated solution.

    Prefers the last \\boxed{...}; falls back to common 'The answer is X' / 'X = '
    patterns; finally the last number in the text.
    """
    boxed = last_boxed_only_string(solution)
    if boxed is not None:
        return remove_boxed(boxed)
    m = re.findall(r"(?:final answer|the answer is|answer:)\s*\$?([^\n.$]+)",
                   solution, flags=re.IGNORECASE)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", solution)
    return nums[-1] if nums else None


# --------------------------------------------------------------------------- #
# Canonical normalization (Hendrycks MATH)
# --------------------------------------------------------------------------- #

def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                try:
                    a, b = substr[0], substr[1]
                except IndexError:
                    return string
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + substr[1:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if string.count("/") != 1:
        return string
    a, b = string.split("/")
    try:
        ia, ib = int(a), int(b)
        if string == f"{ia}/{ib}":
            return f"\\frac{{{ia}}}{{{ib}}}"
        return string
    except (ValueError, TypeError):
        return string


def _remove_right_units(string: str) -> str:
    # remove a trailing "\\text{ ... }" unit annotation
    if "\\text{" in string:
        splits = string.split("\\text{")
        return splits[0].rstrip()
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def normalize_answer(string: str | None) -> str | None:
    """Normalize a LaTeX MATH answer to a canonical string for comparison."""
    if string is None:
        return None
    s = string

    # linebreaks, spaces, \! and \\, decorations
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace(r"\%", "").replace("%", "")
    s = _remove_right_units(s)
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    # "0.5 = " style equalities -> keep rhs
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    # strip surrounding braces / trivial text wrappers
    s = s.replace("\\text{}", "").replace("\\mbox", "")
    return s


# --------------------------------------------------------------------------- #
# sympy equivalence booster (optional)
# --------------------------------------------------------------------------- #

def _sympy_equiv(a: str, b: str) -> bool:
    """True if a and b are symbolically equal. Returns False if sympy can't parse."""
    try:
        import sympy
        from sympy.parsing.sympy_parser import (parse_expr,
                                                standard_transformations,
                                                implicit_multiplication_application)
    except Exception:
        return False

    def _to_expr(s: str):
        # turn the most common LaTeX into sympy-parseable text
        t = s
        t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", t)
        t = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", t)
        t = re.sub(r"\\sqrt(\w)", r"sqrt(\1)", t)
        t = t.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi")
        t = t.replace("{", "(").replace("}", ")").replace("^", "**")
        t = t.replace("\\", "")
        transformations = standard_transformations + (
            implicit_multiplication_application,)
        return parse_expr(t, transformations=transformations, evaluate=True)

    try:
        ea, eb = _to_expr(a), _to_expr(b)
        diff = sympy.simplify(ea - eb)
        return diff == 0
    except Exception:
        return False


def is_equiv(a: str | None, b: str | None) -> bool:
    """Answer-equivalence: canonical string match, then optional sympy fallback."""
    if a is None or b is None:
        return a is None and b is None
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    # cheap numeric equality (e.g. "0.50" vs "0.5", "3.0" vs "3")
    try:
        if abs(float(na) - float(nb)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return _sympy_equiv(a, b)


def grade(generated_solution: str, ground_truth_answer: str) -> dict:
    """Grade a generated solution against the gold answer.

    Returns {pred, gold_norm, correct, gradeable}. ``gradeable`` is False when no
    final answer could be extracted from the generation.
    """
    pred = extract_final_answer(generated_solution)
    return {
        "pred": pred,
        "gold_norm": normalize_answer(ground_truth_answer),
        "correct": bool(is_equiv(pred, ground_truth_answer)),
        "gradeable": pred is not None,
    }
