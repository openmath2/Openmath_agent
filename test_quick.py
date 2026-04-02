"""Quick test to verify agent tests work."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import patch

print("Testing agent imports with mocked ChatOpenAI...")

# Test 1: ReactAgent
print("\n1. Testing ReactAgent...")
try:
    with patch("langchain_openai.ChatOpenAI"):
        from src.agents.react_agent import build_react_agent
        agent = build_react_agent()
        print("   ✓ ReactAgent test PASSED")
except Exception as e:
    print(f"   ✗ ReactAgent test FAILED: {e}")

# Test 2: GeneratorAgent
print("\n2. Testing GeneratorAgent...")
try:
    with patch("langchain_openai.ChatOpenAI"):
        from src.agents.generator_agent import build_generator_agent
        agent = build_generator_agent()
        print("   ✓ GeneratorAgent test PASSED")
except Exception as e:
    print(f"   ✗ GeneratorAgent test FAILED: {e}")

# Test 3: VerifierAgent
print("\n3. Testing VerifierAgent...")
try:
    with patch("langchain_openai.ChatOpenAI"):
        from src.agents.verifier_agent import build_verifier_agent
        agent = build_verifier_agent()
        print("   ✓ VerifierAgent test PASSED")
except Exception as e:
    print(f"   ✗ VerifierAgent test FAILED: {e}")

# Test 4: MultiAgentGraph
print("\n4. Testing MultiAgentGraph...")
try:
    with patch("langchain_openai.ChatOpenAI"):
        from src.agents.multi_agent_graph import build_multi_agent_graph
        graph = build_multi_agent_graph()
        print("   ✓ MultiAgentGraph test PASSED")
except Exception as e:
    print(f"   ✗ MultiAgentGraph test FAILED: {e}")

print("\n" + "=" * 60)
print("Quick test completed!")
print("=" * 60)
