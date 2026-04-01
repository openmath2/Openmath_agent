"""Single ReAct agent that uses SymPy tools to solve math problems."""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.tools import SYMPY_TOOLS  # type: ignore[import]


SYSTEM_PROMPT = """\
You are a precise mathematical reasoning assistant.
Use the available tools to solve problems step by step.
Always verify your answer before returning it.
"""


def build_react_agent(model: str = "gpt-4o", temperature: float = 0.0):
    """Build and return a LangGraph ReAct agent with SymPy tools.

    Args:
        model: OpenAI model name.
        temperature: Sampling temperature.

    Returns:
        Compiled LangGraph agent.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    # TODO: wire tools and system prompt
    agent = create_react_agent(llm, tools=SYMPY_TOOLS)
    return agent
