"""Whitespace-only LaTeX commands cannot change grading or answer identity."""
import pytest

from src.eval.math_grade import answer_key, grade, normalize_answer


@pytest.mark.parametrize("spacing", [r"\ ", r"\,", r"\;", r"\:", r"\quad ", r"\qquad ", r"\!", r"\enspace ", r"\thinspace "])
def test_root_list_spacing_does_not_split_the_vote(spacing):
    pred = "3," + spacing + "5," + spacing + "7"
    assert answer_key(pred) == answer_key("3, 5, 7")
    assert grade("\\boxed{" + pred + "}", "3, 5, 7")["correct"]


def test_spacing_normalization_preserves_different_roots_and_tuple_order():
    assert answer_key(r"3,\ 5,\ 8") != answer_key("3, 5, 7")
    assert answer_key(r"(3,\ 5)") != answer_key("(5, 3)")
    assert normalize_answer(r"\sqrt{5}") == r"\sqrt{5}"
    assert normalize_answer(r"\quadruple") == r"\quadruple"


def test_matrix_row_separator_is_not_a_spacing_command():
    pred = "\\begin{pmatrix}\n-1 & 0 \\\\\n0 & -1\n\\end{pmatrix}"
    gold = r"\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}"
    assert normalize_answer(pred) == normalize_answer(gold)
    assert grade("\\boxed{" + pred + "}", gold)["correct"]
