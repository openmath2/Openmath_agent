"""Smoke tests for agent graph construction (no LLM calls)."""

import pytest
from unittest.mock import patch, MagicMock


class TestReactAgent:
    @patch("langchain_openai.ChatOpenAI")
    def test_build_returns_compiled_graph(self, mock_llm):
        from src.agents.react_agent import build_react_agent
        agent = build_react_agent()
        assert agent is not None

    def test_invoke_structure(self):
        pytest.skip("Requires OpenAI API key — run in integration suite")


class TestGeneratorAgent:
    @patch("langchain_openai.ChatOpenAI")
    def test_build_returns_compiled_graph(self, mock_llm):
        from src.agents.generator_agent import build_generator_agent
        agent = build_generator_agent()
        assert agent is not None


class TestVerifierAgent:
    @patch("langchain_openai.ChatOpenAI")
    def test_build_returns_compiled_graph(self, mock_llm):
        from src.agents.verifier_agent import build_verifier_agent
        agent = build_verifier_agent()
        assert agent is not None


class TestMultiAgentGraph:
    @patch("langchain_openai.ChatOpenAI")
    def test_build_returns_compiled_graph(self, mock_llm):
        from src.agents.multi_agent_graph import build_multi_agent_graph
        graph = build_multi_agent_graph()
        assert graph is not None
