"""Unit tests for SymPy-based tools."""

import pytest
from src.tools.sympy_tools import (
    sympy_solve,
    sympy_simplify,
    sympy_verify,
    sympy_differentiate,
)


def test_solve_quadratic():
    """Test solving a quadratic equation."""
    result = sympy_solve.invoke({"equation": "x**2 - 4 = 0", "variable": "x"})
    # Should return [-2, 2] or [2, -2]
    assert "-2" in result and "2" in result


def test_solve_linear():
    """Test solving a linear equation."""
    result = sympy_solve.invoke({"equation": "2*x + 4 = 0", "variable": "x"})
    # Should return [-2]
    assert "-2" in result


def test_verify_correct():
    """Test verification of two equal expressions."""
    result = sympy_verify.invoke({"expr_a": "(x+1)**2", "expr_b": "x**2 + 2*x + 1"})
    assert result == "VERIFIED"


def test_verify_incorrect():
    """Test verification of two different expressions."""
    result = sympy_verify.invoke({"expr_a": "x**2", "expr_b": "x**3"})
    assert result == "FAILED"


def test_rational_mode():
    """Test that fractions are kept as Rational (not converted to decimals)."""
    result = sympy_solve.invoke({"equation": "2*x + 1 = 0", "variable": "x"})
    # Should contain -1/2 as a Rational, not -0.5
    assert "-1/2" in result or "Rational" in result or "-1/2" in str(result)


def test_sqrt_mode():
    """Test that square roots are kept as sqrt (not converted to decimals)."""
    result = sympy_solve.invoke({"equation": "x**2 - 2 = 0", "variable": "x"})
    # Should contain sqrt(2), not 1.414...
    assert "sqrt(2)" in result or "sqrt" in result


def test_invalid_input():
    """Test handling of invalid input."""
    result = sympy_solve.invoke({"equation": "invalid equation @#$", "variable": "x"})
    # Should return an error message
    assert "Error" in result or "error" in result.lower()
