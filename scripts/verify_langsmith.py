"""Verify LangSmith tracing by running a minimal agent and checking the trace URL."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def check_env() -> list[str]:
    required = ["OPENAI_API_KEY", "LANGCHAIN_API_KEY"]
    return [k for k in required if not os.getenv(k)]


def verify_openai() -> bool:
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=5)
        llm.invoke("Say 'ok'")
        print("[OK] OpenAI API key is valid.")
        return True
    except Exception as e:
        print(f"[FAIL] OpenAI: {e}")
        return False


def verify_langsmith_connection() -> bool:
    try:
        from langsmith import Client
        client = Client()
        projects = list(client.list_projects())
        project_name = os.getenv("LANGCHAIN_PROJECT", "default")
        print(f"[OK] LangSmith connected ({len(projects)} project(s)). Active: {project_name}")
        return True
    except Exception as e:
        print(f"[FAIL] LangSmith connection: {e}")
        return False


def run_trace_smoke_test() -> bool:
    """Run a minimal LangGraph ReAct agent and print the LangSmith trace URL."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent
        from langsmith import Client

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        agent = create_react_agent(llm, tools=[add])

        result = agent.invoke({"messages": [("user", "What is 3 + 7?")]})
        answer = result["messages"][-1].content
        print(f"[OK] Agent answered: {answer}")

        # Fetch the latest run from LangSmith to confirm trace was recorded
        client = Client()
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        runs = list(client.list_runs(project_name=project, limit=1))
        if runs:
            run_id = runs[0].id
            print(f"[OK] Trace recorded: https://smith.langchain.com/o/~/projects/p/{project}/r/{run_id}")
        else:
            print("[WARN] No runs found in project yet — may take a few seconds to appear.")

        return True
    except Exception as e:
        print(f"[FAIL] Smoke test: {e}")
        return False


if __name__ == "__main__":
    print("=== openmath-agents LangSmith verification ===\n")

    missing = check_env()
    if missing:
        print(f"[WARN] Missing env vars: {', '.join(missing)}")
        print("       Copy .env.example to .env and fill in your keys.\n")
        sys.exit(1)

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    print(f"Tracing enabled: {tracing}\n")

    ok = verify_openai()
    ok = verify_langsmith_connection() and ok

    if tracing and ok:
        print("\n--- Running trace smoke test ---")
        ok = run_trace_smoke_test() and ok

    print()
    if ok:
        print("All checks passed.")
        if tracing:
            project = os.getenv("LANGCHAIN_PROJECT", "default")
            print(f"View traces: https://smith.langchain.com → project '{project}'")
        sys.exit(0)
    else:
        print("Some checks failed.")
        sys.exit(1)
