"""Tests for src/analysis/transition_operator_ops.py."""

from __future__ import annotations

from src.analysis.transition_operator_ops import (
    latex_normalize,
    symbolic_operation,
)


def test_latex_normalize_frac_and_times():
    assert "(3)/(4)" in latex_normalize("$\\frac{3}{4}$")
    assert latex_normalize("3 \\times 4").strip() == "3 * 4"
    assert latex_normalize("2 \\cdot 5").strip() == "2 * 5"


def test_verified_addition():
    lab = symbolic_operation("So $12 + 7 = 19$.")
    assert lab.op == "ADD" and lab.verified


def test_verified_multiplication_latex():
    lab = symbolic_operation("We compute $6 \\times 7 = 42$.")
    assert lab.op == "MUL" and lab.verified


def test_verified_division_frac():
    lab = symbolic_operation("$\\frac{12}{4} = 3$")
    assert lab.op == "DIV" and lab.verified


def test_power():
    lab = symbolic_operation("$2^3 = 8$")
    assert lab.op == "POW" and lab.verified


def test_wrong_arithmetic_labeled_but_not_verified():
    lab = symbolic_operation("$12 + 7 = 20$")
    assert lab.op == "ADD" and not lab.verified


def test_subtraction_precedence_over_multiplication():
    # top-level op is the additive one (last applied)
    lab = symbolic_operation("$3 * 4 - 2 = 10$")
    assert lab.op == "SUB" and lab.verified


def test_symbolic_operands_label_without_verify():
    lab = symbolic_operation("This gives $3p + e = 124$.")
    assert lab.op == "ADD" and not lab.verified


def test_no_equation_returns_none():
    assert symbolic_operation("Let me think about this problem.").op is None


def test_unary_minus_not_a_split():
    # "-5 = x" has no binary op on the LHS
    assert symbolic_operation("$-5 = x$").op is None


def test_definition_equation_no_binary_op():
    assert symbolic_operation("That gives me $c = 13.$").op is None
