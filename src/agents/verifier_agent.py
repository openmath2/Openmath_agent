"""Verifier agent: checks the correctness of a candidate solution."""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from typing import TypedDict

from src.tools import SYMPY_TOOLS  # type: ignore[import]


VERIFIER_PROMPT = """\
You are a rigorous mathematical verifier.
Given a problem and a solution candidate, determine whether the solution is correct.
Use symbolic tools (sympy_verify, sympy_solve, sympy_simplify) to confirm equality where possible.
Return a verdict: CORRECT, INCORRECT, or UNCERTAIN with a brief justification.
"""


class VerifierState(TypedDict):
    problem: str
    solution_candidate: str
    verdict: str          # "CORRECT" | "INCORRECT" | "UNCERTAIN"
    justification: str


def build_verifier_agent(model: str | None = None, temperature: float = 0.0):
    """Build the verifier agent subgraph.

    Args:
        model: OpenAI model name. If None, reads from OPENAI_MODEL env var.
        temperature: Low temperature for deterministic verification.

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
    
    # Create ReAct agent with SymPy tools for verification
    react_agent = create_react_agent(
        llm,
        tools=SYMPY_TOOLS,
        prompt=VERIFIER_PROMPT
    )
    
    graph = StateGraph(VerifierState)

    def verify_node(state: VerifierState) -> VerifierState:
        """Verify the solution candidate using the ReAct agent."""
        problem = state["problem"]
        solution = state["solution_candidate"]
        
        # Create verification prompt
        verification_prompt = f"""Problem: {problem}

Solution Candidate: {solution}

Please verify if this solution is correct. Use the available SymPy tools to check the work.
Respond with your verdict (CORRECT, INCORRECT, or UNCERTAIN) and justification."""
        
        # Invoke the ReAct agent
        result = react_agent.invoke({
            "messages": [HumanMessage(content=verification_prompt)]
        })
        
        # Extract verdict and justification from the final message
        messages = result.get("messages", [])
        final_response = ""
        
        if messages:
            final_msg = messages[-1]
            if hasattr(final_msg, "content"):
                final_response = str(final_msg.content)
        
        # Parse verdict from response
        verdict = "UNCERTAIN"
        if "CORRECT" in final_response.upper() and "INCORRECT" not in final_response.upper():
            verdict = "CORRECT"
        elif "INCORRECT" in final_response.upper():
            verdict = "INCORRECT"
        
        return {
            "problem": problem,
            "solution_candidate": solution,
            "verdict": verdict,
            "justification": final_response
        }

    graph.add_node("verify", verify_node)
    graph.set_entry_point("verify")
    graph.set_finish_point("verify")

    return graph.compile()
