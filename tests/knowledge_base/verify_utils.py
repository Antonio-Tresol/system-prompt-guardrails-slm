from pathlib import Path
from typing import Any

from knowledge_base.config.settings import Settings
from knowledge_base.vectordb.chroma_store import ChromaStore
from utils.logging import logger, setup_logging


def get_project_root() -> Path:
    """Returns the absolute path to the project root."""
    return Path(__file__).parent.parent.parent.absolute()


def get_config_path() -> Path:
    """Returns the absolute path to the default config file."""
    return get_project_root() / "knowledge_base" / "config" / "config.yaml"


def setup_verification_env(*, log_file: str | None = None) -> tuple[Settings, ChromaStore]:
    """Sets up the verification environment including settings, logging, and ChromaStore.

    Returns:
        tuple[Settings, ChromaStore]: The initialized settings and Chroma store.
    """
    project_root = get_project_root()
    config_path = get_config_path()

    settings = Settings.load_from_yaml(config_path=str(config_path), project_root=project_root)

    # Use specified log file or default to a generic verification log
    final_log_file = log_file or "verification.log"
    setup_logging(level=settings.logging.level, log_file=final_log_file)

    logger.info(f"Loaded config from {config_path}")

    store = ChromaStore(
        persist_directory=settings.paths.vector_db,
        embeddings_model=settings.embeddings.model,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
    )

    logger.info("Connected to knowledge base")
    return settings, store


def load_all_chunks(store: ChromaStore) -> list[dict[str, Any]]:
    """Loads all chunks and their metadata from the vector store.

    Returns:
        list[dict[str, Any]]: List of dictionaries containing chunk text and metadata.
    """
    results = store.vector_store.get()
    chunks_data = []

    if results["ids"]:
        for i in range(len(results["ids"])):
            meta = results["metadatas"][i]
            text = results["documents"][i]
            # Ensure consistency with keys used in auditors
            meta["document"] = text
            meta["chunk_id"] = results["ids"][i]
            chunks_data.append(meta)

    logger.info(f"Loaded {len(chunks_data)} chunks from database")
    return chunks_data
