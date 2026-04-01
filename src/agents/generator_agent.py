"""Generator agent: produces candidate solutions for a math problem."""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing import TypedDict


GENERATOR_PROMPT = """\
You are a mathematical problem solver.
Given a problem, generate a step-by-step solution candidate.
Be explicit about each reasoning step.
"""


class GeneratorState(TypedDict):
    problem: str
    solution_candidate: str
    reasoning_steps: list[str]


def build_generator_agent(model: str = "gpt-4o", temperature: float = 0.7):
    """Build the generator agent subgraph.

    Args:
        model: OpenAI model name.
        temperature: Higher temperature encourages diverse solutions.

    Returns:
        Compiled LangGraph subgraph.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    graph = StateGraph(GeneratorState)

    # TODO: replace with real LLM node
    def generate_node(state: GeneratorState) -> GeneratorState:
        raise NotImplementedError("Generator node not yet implemented")

    graph.add_node("generate", generate_node)
    graph.set_entry_point("generate")
    graph.set_finish_point("generate")

    return graph.compile()
