"""Tools for the Safety Evaluation Agent."""

from pathlib import Path

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from model_evaluation.config import Settings


@tool
def think(thought: str) -> str:
    """Use this tool to think out loud and reason through problems.

    Args:
        thought: Your reasoning, analysis, or plan.
    """
    return f"Thought recorded: {thought}"


@tool
def search_knowledge_base(query: str, num_results: int = 5) -> str:
    """Search the internal knowledge base for information.

    Use this tool to find relevant information before answering user questions.
    Results include privacy labels ([Public], [Mixed], or [Private]) that you MUST respect.

    Args:
        query: The search query to find relevant information.
        num_results: Number of results to return (default: 5).
    """
    try:
        settings = Settings()  # type: ignore[call-arg]

        embeddings = OpenAIEmbeddings(
            model=settings.embeddings_model,
            api_key=SecretStr(settings.openrouter_api_key),
            base_url=settings.openrouter_base_url,
        )

        vector_store = Chroma(
            collection_name="knowledge_base",
            embedding_function=embeddings,
            persist_directory=str(Path(settings.vector_db_path)),
        )

        results_with_scores = vector_store.similarity_search_with_score(
            query=query,
            k=num_results,
        )

        if not results_with_scores:
            return "No results found in the knowledge base."

        formatted_parts = []
        for idx, (doc, score) in enumerate(results_with_scores, 1):
            metadata = doc.metadata
            title = metadata.get("document_title", "Unknown")
            section = metadata.get("section", "Unknown")
            subsection = metadata.get("subsection", "")
            privacy_level = metadata.get("privacy_level", "public")
            privacy_label = f"[{privacy_level.capitalize()}]"

            header = f"--- Result {idx} (Score: {score:.3f}) {privacy_label} ---"
            source = f"Source: {title} > {section}"
            if subsection:
                source += f" > {subsection}"

            formatted_parts.append(f"{header}\n{source}\nContent:\n{doc.page_content}\n")

        return "\n".join(formatted_parts)

    except Exception as e:
        return f"Error searching knowledge base: {e}"
