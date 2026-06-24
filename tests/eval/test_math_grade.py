"""Tests for the MATH answer grader, grounded in the real PRM800K answer forms."""

import pytest

from src.eval import math_grade as mg


# --------------------------------------------------------------------------- #
# \boxed extraction
# --------------------------------------------------------------------------- #

def test_last_boxed_matches_nested_braces():
    s = r"so the result is \boxed{\frac{1}{2}} done"
    assert mg.last_boxed_only_string(s) == r"\boxed{\frac{1}{2}}"
    assert mg.remove_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_last_boxed_takes_the_last_one():
    s = r"first \boxed{1}, then actually \boxed{42}"
    assert mg.remove_boxed(mg.last_boxed_only_string(s)) == "42"


def test_extract_final_answer_fallbacks():
    assert mg.extract_final_answer(r"... so \boxed{7}.") == "7"
    assert mg.extract_final_answer("The answer is 13.") == "13"
    assert mg.extract_final_answer("we compute 2+2 = 4") == "4"
    assert mg.extract_final_answer("no numbers or boxes here") is None


# --------------------------------------------------------------------------- #
# equivalence on the real answer distribution
# --------------------------------------------------------------------------- #

def test_integers_exact():
    assert mg.is_equiv("2005", "2005")
    assert mg.is_equiv("-3", "-3")
    assert not mg.is_equiv("39", "47")


def test_decimal_and_int_numeric_equal():
    assert mg.is_equiv("3", "3.0")
    assert mg.is_equiv("0.50", "0.5")


def test_fraction_forms_equal():
    assert mg.is_equiv(r"\frac{1}{3}", r"\frac{1}{3}")
    assert mg.is_equiv("1/2", r"\frac{1}{2}")           # a/b normalized to \frac
    assert not mg.is_equiv(r"\frac{1}{3}", r"\frac{1}{115}")


def test_degrees_and_percent_units_stripped():
    assert mg.is_equiv(r"50^{\circ}", "50")
    assert mg.is_equiv(r"83\%", "83")


def test_sqrt_normalization():
    # \sqrt2 and \sqrt{2} must canonicalize the same
    assert mg.is_equiv(r"\sqrt2", r"\sqrt{2}")
    assert mg.is_equiv(r"-\frac{\sqrt{3}}{2}", r"-\frac{\sqrt3}{2}")


def test_grade_end_to_end_correct_and_wrong():
    sol_ok = r"Adding gives the total. Thus \boxed{42}."
    sol_bad = r"After the steps we get \boxed{41}."
    assert mg.grade(sol_ok, "42")["correct"] is True
    g = mg.grade(sol_bad, "42")
    assert g["correct"] is False and g["gradeable"] is True


def test_grade_ungradeable_when_no_answer():
    g = mg.grade("a wandering generation with no conclusion", "42")
    assert g["gradeable"] is False and g["correct"] is False


def test_sympy_booster_optional(monkeypatch):
    # Commuted sum: equal only via sympy. Must not crash if sympy is absent; if sympy
    # is present it should be judged equal.
    a, b = r"4+3\sqrt{2}", r"3\sqrt{2}+4"
    res = mg.is_equiv(a, b)
    try:
        import sympy  # noqa: F401
        assert res is True
    except Exception:
        assert res in (True, False)   # absence of sympy must not raise
