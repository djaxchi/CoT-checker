from src.eval.reprobe_bon_grade import answer_line, grade


def test_answer_line():
    assert answer_line("- Step 1: x\n<Answer>: 72\n<end of response>") == "72"
    assert answer_line("- Step 1: x") is None


def test_gsm8k():
    assert grade("gsm8k", "<Answer>: $1,200", "... #### 1200") == (True, "1200")
    assert grade("gsm8k", "<Answer>: 18%", "#### 15")[0] is False
    assert grade("gsm8k", "- Step 1: no answer", "#### 15") == (False, None)


def test_math():
    gold = r"Solving, we find $a = \boxed{\frac{1}{2}}$."
    assert grade("math", r"<Answer>: $\frac{1}{2}$", gold)[0]
    assert grade("math", r"<Answer>: \boxed{0.5}", gold)[0]
    assert not grade("math", "<Answer>: 2", gold)[0]


def test_strategyqa_and_scienceqa():
    assert grade("strategyqa", "<Answer>: No", "False") == (True, "False")
    assert grade("strategyqa", "<Answer>: Yes, because", "False")[0] is False
    assert grade("scienceqa", "<Answer>: B", "The answer is B. Reasoning: ") == (True, "B")
    assert grade("scienceqa", "<Answer>: (C) the moon", "The answer is B.")[0] is False


def test_calendar():
    gold = "Here is the proposed time: Monday, 14:00 - 14:30 "
    assert grade("calendar", "<Answer>: Monday, 14:00 to 14:30", gold, "on Monday")[0]
    assert grade("calendar", "<Answer>: 14:00 to 14:30", gold, "on Monday")[0]
    assert not grade("calendar", "<Answer>: 14:00 to 14:30", gold, "Monday or Tuesday")[0]
    assert not grade("calendar", "<Answer>: Monday, 9:00 to 9:30", gold, "on Monday")[0]


def test_math_plain_text_answers():
    assert grade("math", "<Answer>: √3/3", r"$\boxed{\frac{\sqrt{3}}{3}}$")[0]
    assert grade("math", "<Answer>: (-∞, 2) ∪ (3, ∞)", r"$\boxed{(-\infty, 2) \cup (3, \infty)}$")[0]
    assert grade("math", "<Answer>: (4/3, -1/3)", r"$\boxed{\left( \frac{4}{3}, -\frac{1}{3} \right)}$")[0]
    assert grade("math", "<Answer>: B", r"$\boxed{\text{(B)}}$")[0]
