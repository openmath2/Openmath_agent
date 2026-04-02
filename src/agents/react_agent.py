"""Single ReAct agent that uses SymPy tools to solve math problems."""

import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.tools import SYMPY_TOOLS  # type: ignore[import]


SYSTEM_PROMPT = """\
You are a precise mathematical reasoning assistant.
Use the available tools to solve problems step by step.
Always verify your answer before returning it.
"""


def build_react_agent(model: str | None = None, temperature: float = 0.0):
    """Build and return a LangGraph ReAct agent with SymPy tools.

    Args:
        model: OpenAI model name. If None, reads from OPENAI_MODEL env var.
        temperature: Sampling temperature.

    Returns:
        Compiled LangGraph agent.
    """
    # Read from environment variables
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "dummy-not-used")
    
    # Create LLM with environment configuration
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key
    )
    
    # Create ReAct agent with SymPy tools and system prompt
    agent = create_react_agent(
        llm,
        tools=SYMPY_TOOLS,
        prompt=SYSTEM_PROMPT
    )
    
    return agent
