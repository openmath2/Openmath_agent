"""Generator agent: produces candidate solutions for a math problem."""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from typing import TypedDict

from src.tools import SYMPY_TOOLS  # type: ignore[import]


GENERATOR_PROMPT = """\
You are a mathematical problem solver.
Given a problem, generate a step-by-step solution candidate.
Be explicit about each reasoning step.
Use the available SymPy tools to solve equations, simplify expressions, and verify your work.
"""


class GeneratorState(TypedDict):
    problem: str
    solution_candidate: str
    reasoning_steps: list[str]


def build_generator_agent(model: str | None = None, temperature: float = 0.7):
    """Build the generator agent subgraph.

    Args:
        model: OpenAI model name. If None, reads from OPENAI_MODEL env var.
        temperature: Higher temperature encourages diverse solutions.

    Returns:
        Compiled LangGraph subgraph.
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
    
    # Create ReAct agent with SymPy tools for generation
    react_agent = create_react_agent(
        llm,
        tools=SYMPY_TOOLS,
        prompt=GENERATOR_PROMPT
    )
    
    graph = StateGraph(GeneratorState)

    def generate_node(state: GeneratorState) -> GeneratorState:
        """Generate a solution candidate using the ReAct agent."""
        problem = state["problem"]
        
        # Invoke the ReAct agent with the problem
        result = react_agent.invoke({
            "messages": [HumanMessage(content=problem)]
        })
        
        # Extract the final answer from messages
        messages = result.get("messages", [])
        solution = ""
        steps = []
        
        for msg in messages:
            if hasattr(msg, "content"):
                steps.append(str(msg.content))
                if msg == messages[-1]:  # Last message is the solution
                    solution = str(msg.content)
        
        return {
            "problem": problem,
            "solution_candidate": solution,
            "reasoning_steps": steps
        }

    graph.add_node("generate", generate_node)
    graph.set_entry_point("generate")
    graph.set_finish_point("generate")

    return graph.compile()
