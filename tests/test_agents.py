"""Smoke tests for agent graph construction (no LLM calls)."""

import pytest
from unittest.mock import patch, MagicMock


class TestReactAgent:
    def test_build_returns_compiled_graph(self):
        with patch("src.agents.react_agent.ChatOpenAI") as mock_llm:
            from src.agents.react_agent import build_react_agent
            agent = build_react_agent()
            assert agent is not None

    def test_invoke_structure(self):
        pytest.skip("Requires OpenAI API key — run in integration suite")


class TestGeneratorAgent:
    def test_build_returns_compiled_graph(self):
        with patch("src.agents.generator_agent.ChatOpenAI") as mock_llm:
            from src.agents.generator_agent import build_generator_agent
            agent = build_generator_agent()
            assert agent is not None


class TestVerifierAgent:
    def test_build_returns_compiled_graph(self):
        with patch("src.agents.verifier_agent.ChatOpenAI") as mock_llm:
            from src.agents.verifier_agent import build_verifier_agent
            agent = build_verifier_agent()
            assert agent is not None


class TestMultiAgentGraph:
    def test_build_returns_compiled_graph(self):
        with patch("src.agents.generator_agent.ChatOpenAI"), \
             patch("src.agents.verifier_agent.ChatOpenAI"):
            from src.agents.multi_agent_graph import build_multi_agent_graph
            graph = build_multi_agent_graph()
            assert graph is not None
