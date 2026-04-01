"""SymPy-based math tools wrapped as LangChain tools."""

from langchain_core.tools import tool
import sympy as sp


@tool
def solve_equation(equation: str, variable: str = "x") -> str:
    """Solve a mathematical equation for a given variable.

    Args:
        equation: Equation as a string (e.g., "x**2 - 4 = 0").
        variable: Variable to solve for (default: "x").

    Returns:
        JSON-serializable string of solutions.
    """
    var = sp.Symbol(variable)
    if "=" in equation:
        lhs, rhs = equation.split("=", 1)
        expr = sp.sympify(lhs.strip()) - sp.sympify(rhs.strip())
    else:
        expr = sp.sympify(equation)
    solutions = sp.solve(expr, var)
    return str(solutions)


@tool
def simplify_expression(expression: str) -> str:
    """Simplify a mathematical expression using SymPy.

    Args:
        expression: Mathematical expression as a string.

    Returns:
        Simplified expression as a string.
    """
    return str(sp.simplify(sp.sympify(expression)))


@tool
def compute_derivative(expression: str, variable: str = "x") -> str:
    """Compute the derivative of an expression with respect to a variable.

    Args:
        expression: Mathematical expression as a string.
        variable: Variable to differentiate with respect to (default: "x").

    Returns:
        Derivative expression as a string.
    """
    var = sp.Symbol(variable)
    return str(sp.diff(sp.sympify(expression), var))


@tool
def compute_integral(expression: str, variable: str = "x") -> str:
    """Compute the indefinite integral of an expression.

    Args:
        expression: Mathematical expression as a string.
        variable: Variable to integrate with respect to (default: "x").

    Returns:
        Integral expression as a string.
    """
    var = sp.Symbol(variable)
    return str(sp.integrate(sp.sympify(expression), var))


@tool
def verify_equality(expr_a: str, expr_b: str) -> bool:
    """Check whether two mathematical expressions are symbolically equal.

    Args:
        expr_a: First expression as a string.
        expr_b: Second expression as a string.

    Returns:
        True if the expressions are symbolically equivalent, False otherwise.
    """
    return sp.simplify(sp.sympify(expr_a) - sp.sympify(expr_b)) == 0


SYMPY_TOOLS = [
    solve_equation,
    simplify_expression,
    compute_derivative,
    compute_integral,
    verify_equality,
]
