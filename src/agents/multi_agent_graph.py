"""Top-level LangGraph that orchestrates generator and verifier agents."""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

from .generator_agent import build_generator_agent, GeneratorState
from .verifier_agent import build_verifier_agent, VerifierState


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


def build_multi_agent_graph(model: str | None = None):
    """Compose generator and verifier into a single orchestration graph.

    This implements a Supervisor pattern where:
    1. Generator creates a solution candidate
    2. Verifier checks the solution
    3. If incorrect/uncertain and retries remain, loop back to generator
    4. If correct or max retries reached, finish

    Args:
        model: OpenAI model name. If None, reads from OPENAI_MODEL env var.

    Returns:
        Compiled LangGraph.
    """
    # Read from environment variables
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Build sub-agents
    generator = build_generator_agent(model=model)
    verifier = build_verifier_agent(model=model)

    graph = StateGraph(MultiAgentState)

    def generator_node(state: MultiAgentState) -> MultiAgentState:
        """Invoke generator agent to create a solution."""
        # Prepare generator input
        gen_input: GeneratorState = {
            "problem": state["problem"],
            "solution_candidate": "",
            "reasoning_steps": []
        }
        
        # Invoke generator
        gen_result = generator.invoke(gen_input)
        
        # Update state with generator output
        return {
            "problem": state["problem"],
            "solution_candidate": gen_result["solution_candidate"],
            "reasoning_steps": gen_result["reasoning_steps"],
            "verdict": state.get("verdict", ""),
            "justification": state.get("justification", ""),
            "attempt": state.get("attempt", 0) + 1
        }

    def verifier_node(state: MultiAgentState) -> MultiAgentState:
        """Invoke verifier agent to check the solution."""
        # Prepare verifier input
        ver_input: VerifierState = {
            "problem": state["problem"],
            "solution_candidate": state["solution_candidate"],
            "verdict": "",
            "justification": ""
        }
        
        # Invoke verifier
        ver_result = verifier.invoke(ver_input)
        
        # Update state with verifier output
        return {
            "problem": state["problem"],
            "solution_candidate": state["solution_candidate"],
            "reasoning_steps": state["reasoning_steps"],
            "verdict": ver_result["verdict"],
            "justification": ver_result["justification"],
            "attempt": state["attempt"]
        }

    # Add nodes
    graph.add_node("generate", generator_node)
    graph.add_node("verify", verifier_node)

    # Set entry point
    graph.set_entry_point("generate")

    # Add edges
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "generate": "generate",
            END: END
        }
    )

    return graph.compile()
