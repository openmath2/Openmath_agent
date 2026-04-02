"""SymPy-based math tools wrapped as LangChain tools."""

from langchain_core.tools import tool
import sympy as sp
from sympy import Rational, sqrt


@tool
def sympy_solve(equation: str, variable: str = "x") -> str:
    """Solve a mathematical equation for a given variable.
    
    Fractions are kept as Rational, roots are kept as sqrt (not converted to decimals).

    Args:
        equation: Equation as a string (e.g., "x**2 - 4 = 0" or "2*x + 1 = 0").
        variable: Variable to solve for (default: "x").

    Returns:
        String representation of solutions with Rational and sqrt preserved.
    """
    try:
        var = sp.Symbol(variable)
        if "=" in equation:
            lhs, rhs = equation.split("=", 1)
            expr = sp.sympify(lhs.strip()) - sp.sympify(rhs.strip())
        else:
            expr = sp.sympify(equation)
        
        solutions = sp.solve(expr, var)
        
        # Convert solutions to use Rational and sqrt explicitly
        result = []
        for sol in solutions:
            result.append(sol)
        
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def sympy_simplify(expression: str) -> str:
    """Simplify a mathematical expression using SymPy.

    Args:
        expression: Mathematical expression as a string.

    Returns:
        Simplified expression as a string.
    """
    try:
        return str(sp.simplify(sp.sympify(expression)))
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def sympy_verify(expr_a: str, expr_b: str) -> str:
    """Check whether two mathematical expressions are symbolically equal.

    Args:
        expr_a: First expression as a string.
        expr_b: Second expression as a string.

    Returns:
        "VERIFIED" if the expressions are symbolically equivalent, "FAILED" otherwise.
    """
    try:
        diff = sp.simplify(sp.sympify(expr_a) - sp.sympify(expr_b))
        if diff == 0:
            return "VERIFIED"
        else:
            return "FAILED"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def sympy_differentiate(expression: str, variable: str = "x") -> str:
    """Compute the derivative of an expression with respect to a variable.

    Args:
        expression: Mathematical expression as a string.
        variable: Variable to differentiate with respect to (default: "x").

    Returns:
        Derivative expression as a string.
    """
    try:
        var = sp.Symbol(variable)
        return str(sp.diff(sp.sympify(expression), var))
    except Exception as e:
        return f"Error: {str(e)}"


SYMPY_TOOLS = [
    sympy_solve,
    sympy_simplify,
    sympy_verify,
    sympy_differentiate,
]
