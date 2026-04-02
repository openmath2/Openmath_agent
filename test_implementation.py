"""Quick test script to verify the implementation."""

import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8317/v1")
os.environ.setdefault("OPENAI_MODEL", "gemini-2.5-pro")

print("=" * 60)
print("Testing OpenMath Agents Implementation")
print("=" * 60)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from src.agents.react_agent import build_react_agent
    from src.agents.generator_agent import build_generator_agent
    from src.agents.verifier_agent import build_verifier_agent
    from src.agents.multi_agent_graph import build_multi_agent_graph
    from src.evaluation.dataset import get_middle_school_benchmark
    from src.tools import SYMPY_TOOLS
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Check SymPy tools
print("\n2. Testing SymPy tools...")
try:
    print(f"   ✓ Found {len(SYMPY_TOOLS)} SymPy tools:")
    for tool in SYMPY_TOOLS:
        print(f"     - {tool.name}")
except Exception as e:
    print(f"   ✗ SymPy tools check failed: {e}")

# Test 3: Build ReAct agent
print("\n3. Building ReAct agent...")
try:
    from unittest.mock import patch, MagicMock
    with patch("langchain_openai.ChatOpenAI") as mock_llm:
        mock_llm.return_value = MagicMock()
        agent = build_react_agent()
        print(f"   ✓ ReAct agent built successfully: {type(agent)}")
except Exception as e:
    print(f"   ✗ ReAct agent build failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Build Generator agent
print("\n4. Building Generator agent...")
try:
    with patch("langchain_openai.ChatOpenAI") as mock_llm:
        mock_llm.return_value = MagicMock()
        agent = build_generator_agent()
        print(f"   ✓ Generator agent built successfully: {type(agent)}")
except Exception as e:
    print(f"   ✗ Generator agent build failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Build Verifier agent
print("\n5. Building Verifier agent...")
try:
    with patch("langchain_openai.ChatOpenAI") as mock_llm:
        mock_llm.return_value = MagicMock()
        agent = build_verifier_agent()
        print(f"   ✓ Verifier agent built successfully: {type(agent)}")
except Exception as e:
    print(f"   ✗ Verifier agent build failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Build Multi-Agent graph
print("\n6. Building Multi-Agent graph...")
try:
    with patch("langchain_openai.ChatOpenAI") as mock_llm:
        mock_llm.return_value = MagicMock()
        graph = build_multi_agent_graph()
        print(f"   ✓ Multi-Agent graph built successfully: {type(graph)}")
except Exception as e:
    print(f"   ✗ Multi-Agent graph build failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Load middle school benchmark
print("\n7. Loading middle school benchmark...")
try:
    dataset = get_middle_school_benchmark()
    print(f"   ✓ Loaded {len(dataset)} problems")
    print(f"\n   Sample problems:")
    for i, problem in enumerate(dataset.problems[:3]):
        print(f"     {i+1}. [{problem.id}] {problem.problem}")
        print(f"        Answer: {problem.answer}")
        print(f"        Tags: {', '.join(problem.tags)}")
except Exception as e:
    print(f"   ✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
