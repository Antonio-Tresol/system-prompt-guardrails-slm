from langchain_core.tools import tool, BaseTool


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for information."""
    return f"Mock search result for: {query}"


def get_knowledge_base_tool() -> BaseTool:
    """Returns the knowledge base tool."""
    return search_knowledge_base
