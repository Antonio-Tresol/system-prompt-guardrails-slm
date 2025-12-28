from pathlib import Path
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge_base.config.settings import Settings
from knowledge_base.ingest.chunkers import Chunk, chunk_document
from knowledge_base.ingest.loaders import load_markdown, load_pdf
from knowledge_base.ingest.metadata_extractor import extract_metadata
from knowledge_base.ingest.privacy_detector import PrivacyResult, detect_privacy
from knowledge_base.schemas.chunk_metadata import ChunkMetadata
from knowledge_base.utils.file_tracker import FileTracker
from knowledge_base.vectordb.chroma_store import ChromaStore
from utils.logging import logger


class IngestionState(TypedDict):
    """State for document ingestion pipeline."""

    file_path: str
    document: Any | None  # noqa: ANN401
    chunks: list[Chunk]
    privacy_results: list[PrivacyResult]
    metadatas: list[ChunkMetadata]
    errors: list[str]
    settings: Settings
    chroma_store: ChromaStore
    file_tracker: FileTracker
    model: ChatOpenAI


def load_document_node(state: IngestionState) -> IngestionState:
    """Load document from file path."""
    file_path = Path(state["file_path"])

    try:
        match file_path.suffix.lower():
            case ".md":
                doc = load_markdown(file_path)
            case ".pdf":
                doc = load_pdf(file_path)
            case _:
                state["errors"].append(f"Unsupported file type: {file_path.suffix}")
                return state

        state["document"] = doc
        logger.info(f"Successfully loaded {file_path}")
    except Exception as e:
        state["errors"].append(f"Failed to load {file_path}: {e}")
        logger.error(f"Failed to load {file_path}: {e}")

    return state


def chunk_document_node(state: IngestionState) -> IngestionState:
    """Chunk the loaded document."""
    if state["document"] is None:
        return state

    try:
        # Read raw text for precise line mapping in Markdown
        raw_text = None
        file_path = Path(state["file_path"])
        if file_path.suffix.lower() == ".md":
            try:
                raw_text = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to read raw text for {file_path}: {e}")

        chunks = chunk_document(
            doc=state["document"],
            max_chunk_size=state["settings"].chunking.max_chunk_size,
            min_chunk_size=state["settings"].chunking.min_chunk_size,
            raw_text=raw_text,
        )
        state["chunks"] = chunks
        logger.info(f"Created {len(chunks)} chunks")
    except Exception as e:
        state["errors"].append(f"Failed to chunk document: {e}")
        logger.error(f"Failed to chunk document: {e}")

    return state


def detect_privacy_node(state: IngestionState) -> IngestionState:
    """Detect privacy level for each chunk."""
    if not state["chunks"]:
        return state

    privacy_results: list[PrivacyResult] = []

    for chunk in state["chunks"]:
        try:
            result = detect_privacy(
                chunk=chunk,
                keywords=state["settings"].private_keywords,
                model=state["model"],
            )
            privacy_results.append(result)
        except Exception as e:
            logger.warning(f"Privacy detection failed for chunk, using default: {e}")
            privacy_results.append(
                PrivacyResult(
                    has_private_info=False,
                    privacy_level="public",
                    reasoning="Error in detection",
                ),
            )

    state["privacy_results"] = privacy_results
    logger.info(f"Detected privacy for {len(privacy_results)} chunks")

    return state


def extract_metadata_node(state: IngestionState) -> IngestionState:
    """Extract metadata for each chunk."""
    if not state["chunks"] or not state["privacy_results"]:
        return state

    file_path = Path(state["file_path"])
    document_title = file_path.stem.replace("_", " ")

    metadatas: list[ChunkMetadata] = []

    for idx, (chunk, privacy_result) in enumerate(
        zip(
            state["chunks"],
            state["privacy_results"],
            strict=False,
        ),
    ):
        try:
            metadata = extract_metadata(
                chunk=chunk,
                privacy_result=privacy_result,
                chunk_idx=idx,
                document_title=document_title,
                source_file=str(file_path),
                start_page=chunk.start_page,  # Map start_page
                end_page=chunk.end_page,  # Map end_page
                page_number=chunk.start_page,  # Use start_page as page_number for simplicity
            )
            metadatas.append(metadata)
        except Exception as e:
            logger.error(f"Failed to extract metadata for chunk {idx}: {e}")

    state["metadatas"] = metadatas
    logger.info(f"Extracted metadata for {len(metadatas)} chunks")

    return state


def embed_and_store_node(state: IngestionState) -> IngestionState:
    """Embed chunks and store in ChromaDB."""
    if not state["chunks"] or not state["metadatas"]:
        return state

    try:
        texts = [chunk.text for chunk in state["chunks"]]
        ids = [f"{state['file_path']}_{idx}" for idx in range(len(state["chunks"]))]

        success_count = state["chroma_store"].add_chunks(
            texts=texts,
            metadatas=state["metadatas"],
            ids=ids,
        )

        if success_count > 0:
            file_path = Path(state["file_path"])
            state["file_tracker"].track_file(path=file_path, timestamp=file_path.stat().st_mtime)
            logger.info(f"Successfully stored {success_count} chunks for {state['file_path']}")
    except Exception as e:
        state["errors"].append(f"Failed to store chunks: {e}")
        logger.error(f"Failed to store chunks: {e}")

    return state


def create_ingestion_pipeline(
    *,
    settings: Settings,
    chroma_store: ChromaStore,
    file_tracker: FileTracker,
    model: ChatOpenAI,
) -> CompiledStateGraph:
    """Create LangGraph ingestion pipeline.

    Args:
        settings: Configuration settings.
        chroma_store: ChromaDB store instance.
        file_tracker: File tracker instance.
        model: model client for privacy detection.

    Returns:
        Compiled LangGraph pipeline.
    """
    workflow: StateGraph = StateGraph(IngestionState)

    workflow.add_node("load_document", load_document_node)  # type: ignore[arg-type]
    workflow.add_node("chunk_document", chunk_document_node)  # type: ignore[arg-type]
    workflow.add_node("detect_privacy", detect_privacy_node)  # type: ignore[arg-type]
    workflow.add_node("extract_metadata", extract_metadata_node)  # type: ignore[arg-type]
    workflow.add_node("embed_and_store", embed_and_store_node)  # type: ignore[arg-type]

    workflow.add_edge(START, "load_document")
    workflow.add_edge("load_document", "chunk_document")
    workflow.add_edge("chunk_document", "detect_privacy")
    workflow.add_edge("detect_privacy", "extract_metadata")
    workflow.add_edge("extract_metadata", "embed_and_store")
    workflow.add_edge("embed_and_store", END)

    return workflow.compile()
