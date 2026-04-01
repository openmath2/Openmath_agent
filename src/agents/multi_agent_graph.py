"""Top-level LangGraph that orchestrates generator and verifier agents."""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

from .generator_agent import build_generator_agent
from .verifier_agent import build_verifier_agent


MAX_RETRIES = 3


class MultiAgentState(TypedDict):
    problem: str
    solution_candidate: str
    reasoning_steps: list[str]
    verdict: str
    justification: str
    attempt: int


def route_after_verify(state: MultiAgentState) -> Literal["generate", "__end__"]:
    """Route back to generator on INCORRECT/UNCERTAIN, else finish."""
    if state["verdict"] == "CORRECT":
        return END
    if state["attempt"] >= MAX_RETRIES:
        return END
    return "generate"


def build_multi_agent_graph():
    """Compose generator and verifier into a single orchestration graph.

    Returns:
        Compiled LangGraph.
    """
    generator = build_generator_agent()
    verifier = build_verifier_agent()

    graph = StateGraph(MultiAgentState)

    # TODO: replace with real subgraph nodes and routing logic
    def placeholder_node(state: MultiAgentState) -> MultiAgentState:
        raise NotImplementedError("Multi-agent graph not yet implemented")

    graph.add_node("generate", placeholder_node)
    graph.set_entry_point("generate")
    graph.set_finish_point("generate")

    return graph.compile()
