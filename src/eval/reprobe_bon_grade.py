"""Final-answer grading for ReProbe's released best-of-N pools (reprobe_comparison_v2, B1).

ReProbe (arXiv 2511.06209, Table 3) grades GSM8K by exact match and every other set with
DeepSeek-R1; the pools (`JingweiNi/test_*_Qwen3-8B_texts_uhead_prm`) carry the gold
answer but no correctness. These are our programmatic graders for the five sets where a
final answer can be checked without a judge. ProofNet (proofs), Trip and Meeting
planning (free-form itineraries) need a judge and have no grader here.

Every grader reads the reply's `<Answer>:` line (ReProbe's format) and returns
(correct, key): `key` is a normalised answer identity for majority voting, None when no
answer could be read (then `correct` is False).
"""

from __future__ import annotations

import re

from src.eval.math_grade import answer_key, is_equiv, last_boxed_only_string, remove_boxed

GRADED = ("math", "gsm8k", "strategyqa", "scienceqa", "calendar")
_ANSWER = re.compile(r"<Answer>:\s*(.*)", re.S)
_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def answer_line(reply: str) -> str | None:
    m = _ANSWER.search(reply)
    if m is None:
        return None
    a = m.group(1).split("<end of response>")[0].strip()
    return a or None


def _last_number(s: str) -> float | None:
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", "").replace("$", ""))
    return float(m[-1]) if m else None


def grade_gsm8k(reply: str, gold: str) -> tuple[bool, str | None]:
    a = answer_line(reply)
    p, g = (None if a is None else _last_number(a)), _last_number(gold.split("####")[-1])
    if p is None:
        return False, None
    return g is not None and abs(p - g) < 1e-6, f"{p:g}"


_UNICODE = (("√", "\\sqrt"), ("∞", "\\infty"), ("∪", "\\cup"), ("π", "\\pi"), ("°", "^\\circ"),
            ("−", "-"), ("×", "\\times"), ("≤", "\\le"), ("≥", "\\ge"))


def _latexish(s: str) -> str:
    """Plain-text answers in the gold's LaTeX spelling: unicode symbols, \\left/\\right,
    \\text{...} wrappers, and √x as \\sqrt{x}."""
    for u, t in _UNICODE:
        s = s.replace(u, t)
    s = re.sub(r"\\sqrt\s*(\d+|[a-zA-Z])", r"\\sqrt{\1}", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{\s*\(?([^{}]*?)\)?\s*\}", r"\1", s)
    return s.strip()


def grade_math(reply: str, gold_solution: str) -> tuple[bool, str | None]:
    a = answer_line(reply)
    if a is None:
        return False, None
    a = a.strip().strip("$").strip().rstrip(".")
    boxed = last_boxed_only_string(a)
    if boxed is not None:
        a = remove_boxed(boxed)
    a = _latexish(a)
    gold = _latexish(remove_boxed(last_boxed_only_string(gold_solution)) or "")
    return _equiv(a, gold), answer_key(a)


def _braced(s: str, i: int) -> tuple[str, int] | None:
    """The {...} group opening at s[i] (nested braces allowed) and the index after it."""
    if i >= len(s) or s[i] != "{":
        return None
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
    return None


def _frac_to_slash(s: str) -> str:
    """\\frac{A}{B} -> (A)/(B) with nested braces, which math_grade's regex does not take."""
    out, i = "", 0
    while i < len(s):
        if s.startswith("\\frac", i) or s.startswith("\\dfrac", i):
            j = i + (5 if s.startswith("\\frac", i) else 6)
            num = _braced(s, j)
            den = _braced(s, num[1]) if num else None
            if den:
                out += f"({_frac_to_slash(num[0])})/({_frac_to_slash(den[0])})"
                i = den[1]
                continue
        out += s[i]
        i += 1
    return out


def _parts(s: str) -> list[str] | None:
    """The elements of a parenthesised tuple at top level, None if s is not one."""
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")) or "\\cup" in s:
        return None
    parts, depth, cur = [], 0, ""
    for ch in s[1:-1]:
        depth += {"(": 1, "{": 1, "[": 1, ")": -1, "}": -1, "]": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur] if parts else None


def _equiv(a: str, gold: str) -> bool:
    if is_equiv(a, gold):
        return True
    pa, pg = _parts(a), _parts(gold)
    if pa is not None and pg is not None:
        return len(pa) == len(pg) and all(_equiv(x.strip(), y.strip()) for x, y in zip(pa, pg))
    fa, fg = _frac_to_slash(a), _frac_to_slash(gold)
    return (fa, fg) != (a, gold) and bool(is_equiv(fa, fg))


def grade_strategyqa(reply: str, gold: str) -> tuple[bool, str | None]:
    a = answer_line(reply)
    if a is None:
        return False, None
    w = re.findall(r"\b(yes|no|true|false)\b", a.lower())
    if not w:
        return False, None
    p = w[0] in ("yes", "true")
    return p == (gold.strip().lower() in ("true", "yes")), str(p)


def grade_scienceqa(reply: str, gold: str) -> tuple[bool, str | None]:
    a = answer_line(reply)
    g = re.search(r"answer is\s*\(?([A-E])\b", gold)
    if a is None or g is None:
        return False, None
    m = re.match(r"\s*\(?([A-E])\b", a) or re.search(r"\b(?:option|answer)\s*\(?([A-E])\b", a, re.I)
    if m is None:
        return False, None
    return m.group(1) == g.group(1), m.group(1)


def _times(s: str) -> list[tuple[int, int]]:
    return [(int(h), int(mn)) for h, mn in re.findall(r"\b(\d{1,2}):(\d{2})\b", s)]


def grade_calendar(reply: str, gold: str, question: str = "") -> tuple[bool, str | None]:
    """Start and end times must match the gold slot; the day must match too when the
    reply names one, or when the question offers more than one day."""
    a = answer_line(reply)
    gt, pt = _times(gold), (None if a is None else _times(a))
    if not pt or len(pt) < 2 or len(gt) < 2:
        return False, None
    gday = next((d for d in _DAYS if d in gold.lower()), None)
    pday = next((d for d in _DAYS if d in a.lower()), None)
    days_in_q = {d for d in _DAYS if d in question.lower()}
    day_ok = pday == gday if (pday is not None or len(days_in_q) > 1) else True
    key = f"{pday or ''} {pt[0][0]:02d}:{pt[0][1]:02d}-{pt[1][0]:02d}:{pt[1][1]:02d}"
    return day_ok and pt[:2] == gt[:2], key


def grade(set_name: str, reply: str, gold: str, question: str = "") -> tuple[bool, str | None]:
    if set_name == "gsm8k":
        return grade_gsm8k(reply, gold)
    if set_name == "math":
        return grade_math(reply, gold)
    if set_name == "strategyqa":
        return grade_strategyqa(reply, gold)
    if set_name == "scienceqa":
        return grade_scienceqa(reply, gold)
    if set_name == "calendar":
        return grade_calendar(reply, gold, question)
    raise ValueError(f"no programmatic grader for {set_name}")
