"""Unit tests for SymPy-based tools."""

import pytest
from src.tools.sympy_tools import (
    solve_equation,
    simplify_expression,
    compute_derivative,
    compute_integral,
    verify_equality,
)


class TestSolveEquation:
    def test_linear(self):
        result = solve_equation.invoke({"equation": "2*x + 4 = 0", "variable": "x"})
        assert "-2" in result

    def test_quadratic(self):
        result = solve_equation.invoke({"equation": "x**2 - 4 = 0", "variable": "x"})
        assert "2" in result and "-2" in result


class TestSimplifyExpression:
    def test_basic(self):
        result = simplify_expression.invoke({"expression": "x**2 + 2*x + 1"})
        assert "(x + 1)**2" in result or "x**2 + 2*x + 1" in result


class TestComputeDerivative:
    def test_polynomial(self):
        result = compute_derivative.invoke({"expression": "x**3", "variable": "x"})
        assert "3*x**2" in result or "3x^2" in result


class TestComputeIntegral:
    def test_polynomial(self):
        result = compute_integral.invoke({"expression": "x**2", "variable": "x"})
        assert "x**3" in result


class TestVerifyEquality:
    def test_equal(self):
        assert verify_equality.invoke({"expr_a": "(x+1)**2", "expr_b": "x**2+2*x+1"})

    def test_not_equal(self):
        assert not verify_equality.invoke({"expr_a": "x**2", "expr_b": "x**3"})
