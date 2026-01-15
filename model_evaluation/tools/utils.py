"""Utility tools for the agent."""

from langchain_core.tools import Tool


def think_func(thought: str) -> str:
    """Record a thought."""
    return f"Thought recorded: {thought}"


def get_think_tool() -> Tool:
    """Create a Thinking tool.

    This tool allows the model to output reasoning traces or "thoughts"
    explicitly before taking actions or verifying retrieval results.
    """
    return Tool(
        name="think",
        func=think_func,
        description=(
            "Use this tool to think out loud. Input your reasoning, plan, "
            "or analysis of the retrieved information here. This helps track your logic."
        ),
    )
