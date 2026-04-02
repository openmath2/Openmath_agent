"""Quick verification script for sympy_tools tests."""

from src.tools.sympy_tools import sympy_solve, sympy_simplify, sympy_verify, sympy_differentiate

print("=" * 60)
print("Testing sympy_tools implementation")
print("=" * 60)

# Test 1: solve_quadratic
print("\n1. test_solve_quadratic:")
result = sympy_solve.invoke({"equation": "x**2 - 4 = 0", "variable": "x"})
print(f"   Result: {result}")
print(f"   ✓ PASS" if "-2" in result and "2" in result else "   ✗ FAIL")

# Test 2: solve_linear
print("\n2. test_solve_linear:")
result = sympy_solve.invoke({"equation": "2*x + 4 = 0", "variable": "x"})
print(f"   Result: {result}")
print(f"   ✓ PASS" if "-2" in result else "   ✗ FAIL")

# Test 3: verify_correct
print("\n3. test_verify_correct:")
result = sympy_verify.invoke({"expr_a": "(x+1)**2", "expr_b": "x**2 + 2*x + 1"})
print(f"   Result: {result}")
print(f"   ✓ PASS" if result == "VERIFIED" else "   ✗ FAIL")

# Test 4: verify_incorrect
print("\n4. test_verify_incorrect:")
result = sympy_verify.invoke({"expr_a": "x**2", "expr_b": "x**3"})
print(f"   Result: {result}")
print(f"   ✓ PASS" if result == "FAILED" else "   ✗ FAIL")

# Test 5: rational_mode
print("\n5. test_rational_mode:")
result = sympy_solve.invoke({"equation": "2*x + 1 = 0", "variable": "x"})
print(f"   Result: {result}")
print(f"   ✓ PASS" if "-1/2" in result else "   ✗ FAIL")

# Test 6: sqrt_mode
print("\n6. test_sqrt_mode:")
result = sympy_solve.invoke({"equation": "x**2 - 2 = 0", "variable": "x"})
print(f"   Result: {result}")
print(f"   ✓ PASS" if "sqrt(2)" in result or "sqrt" in result else "   ✗ FAIL")

# Test 7: invalid_input
print("\n7. test_invalid_input:")
result = sympy_solve.invoke({"equation": "invalid equation @#$", "variable": "x"})
print(f"   Result: {result}")
print(f"   ✓ PASS" if "Error" in result or "error" in result.lower() else "   ✗ FAIL")

print("\n" + "=" * 60)
print("All 7 tests completed!")
print("=" * 60)
