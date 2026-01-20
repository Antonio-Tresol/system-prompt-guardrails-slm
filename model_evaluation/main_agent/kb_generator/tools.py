"""Tools for the Knowledge Base Generator Agent."""

from langchain.tools import ToolRuntime, tool

from model_evaluation.main_agent.kb_generator.schemas import GeneratorContext


@tool
def think(
    thought: str,
    *,
    runtime: ToolRuntime[GeneratorContext],
) -> str:
    """Use this tool to plan and review your document generation.

    Use this to:
    1. Plan what documents to generate based on the query
    2. Review what you have previously generated (check your message history)
    3. Ensure coherence with previous generations before responding

    Args:
        thought: Your reasoning, planning, or review of generated content.
        runtime: Injected by LangChain, provides access to context.
    """
    privacy_mode = "PRIVATE" if runtime.context.include_private_info else "PUBLIC"
    return f"[{privacy_mode} mode] Thought recorded: {thought}"
