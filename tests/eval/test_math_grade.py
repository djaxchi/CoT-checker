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


# --------------------------------------------------------------------------- #
# forms the Qwen3-8B (Instruct) MATH-500 pool writes, found by listing problems
# where 8+ of 10 samples agreed on an answer the grader marked wrong
# (instruct_arm_v1). Each pair is (model answer, MATH-500 gold).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pred,gold", [
    (r"\frac{17}{50}", r"\dfrac{17}{50}"),          # math500_00162
    (r"\dfrac{13}{6}", r"\frac{13}{6}"),            # math500_00331
    ("10080", r"10,\!080"),                         # math500_00198
    ("32348", r"\$32,\!348"),                       # math500_00242
    ("11111111100", r"11,\! 111,\! 111,\! 100"),    # math500_00217
    ("58500", "58,500"),                            # math500_00343
    ("864", r"864 \mbox{ inches}^2"),               # math500_00257
    ("15", r"15\mbox{ cm}^2"),                      # math500_00467
    ("4210_5", "4210_{5}"),                         # math500_00127
    ("[-2, 7]", r"x \in [-2,7]"),                   # math500_00383
    ("1, -2", "-2,1"),                              # math500_00456
])
def test_instruct_pool_forms_are_equivalent(pred, gold):
    assert mg.is_equiv(pred, gold)


@pytest.mark.parametrize("pred,gold", [
    ("3", r"\frac{13}{4}"),                         # math500_00401, a real error
    ("1", "501"),                                   # math500_00080, a real error
    ("21", "28"),                                   # math500_00303, a real error
    ("(1, 2)", "(2, 1)"),                           # an ordered pair stays ordered
    ("1,2", "1,3"),
    ("1,000", "1"),                                 # a thousands group is not a list
])
def test_new_normalisations_do_not_overmatch(pred, gold):
    assert not mg.is_equiv(pred, gold)
