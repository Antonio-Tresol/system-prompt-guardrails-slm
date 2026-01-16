"""Tools for the Safety Evaluation Agent."""

from pathlib import Path

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.config import get_stream_writer
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
        # Get stream writer for emitting custom events (may not exist outside LangGraph context)
        try:
            writer = get_stream_writer()
        except Exception:
            writer = None

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

        # Emit search start event
        if writer:
            writer(
                {
                    "event": "search_start",
                    "query": query,
                    "num_results": num_results,
                }
            )

        results_with_scores = vector_store.similarity_search_with_score(
            query=query,
            k=num_results,
        )

        if not results_with_scores:
            if writer:
                writer({"event": "search_complete", "chunks_found": 0})
            return "No results found in the knowledge base."

        # Emit metadata for each chunk found
        if writer:
            chunks_metadata = []
            for idx, (doc, score) in enumerate(results_with_scores, 1):
                metadata = doc.metadata
                chunk_info = {
                    "rank": idx,
                    "score": round(score, 3),
                    "document_title": metadata.get("document_title", "Unknown"),
                    "section": metadata.get("section", "Unknown"),
                    "subsection": metadata.get("subsection", ""),
                    "privacy_level": metadata.get("privacy_level", "public"),
                    "content_preview": doc.page_content[:100] + "..."
                    if len(doc.page_content) > 100
                    else doc.page_content,
                }
                chunks_metadata.append(chunk_info)

            writer(
                {
                    "event": "search_complete",
                    "chunks_found": len(results_with_scores),
                    "chunks": chunks_metadata,
                }
            )

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
