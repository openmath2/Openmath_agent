"""Verifier agent: checks the correctness of a candidate solution."""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing import TypedDict

from src.tools import verify_equality  # type: ignore[import]


VERIFIER_PROMPT = """\
You are a rigorous mathematical verifier.
Given a problem and a solution candidate, determine whether the solution is correct.
Use symbolic tools to confirm equality where possible.
Return a verdict: CORRECT, INCORRECT, or UNCERTAIN with a brief justification.
"""


class VerifierState(TypedDict):
    problem: str
    solution_candidate: str
    verdict: str          # "CORRECT" | "INCORRECT" | "UNCERTAIN"
    justification: str


def build_verifier_agent(model: str = "gpt-4o", temperature: float = 0.0):
    """Build the verifier agent subgraph.

    Args:
        model: OpenAI model name.
        temperature: Low temperature for deterministic verification.

    Returns:
        Compiled LangGraph subgraph.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    graph = StateGraph(VerifierState)

    # TODO: replace with real LLM node
    def verify_node(state: VerifierState) -> VerifierState:
        raise NotImplementedError("Verifier node not yet implemented")

    graph.add_node("verify", verify_node)
    graph.set_entry_point("verify")
    graph.set_finish_point("verify")

    return graph.compile()
