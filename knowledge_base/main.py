from pathlib import Path

from langchain_openai import ChatOpenAI

from knowledge_base.config.settings import Settings
from knowledge_base.ingest.pipeline import create_ingestion_pipeline
from knowledge_base.utils.file_tracker import FileTracker
from knowledge_base.vectordb.chroma_store import ChromaStore
from utils.logging import logger, setup_logging


def main() -> None:
    """Build and populate vector database."""
    settings = Settings.load_from_yaml()
    setup_logging(level=settings.logging.level, log_file="knowledge_base.log")

    logger.info("Starting knowledge base ingestion")

    source_docs_path = Path(settings.paths.source_documents)

    file_tracker = FileTracker(settings.paths.file_tracker)

    chroma_store = ChromaStore(
        persist_directory=settings.paths.vector_db,
        embeddings_model=settings.embeddings.model,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
    )

    if chroma_store.collection_exists():
        unprocessed = file_tracker.get_unprocessed_files(source_docs_path)
        if not unprocessed:
            logger.info("✅ Vector DB is up-to-date")
            return
        logger.info(f"Found {len(unprocessed)} new/modified documents")
    else:
        logger.info("Creating new vector database")
        unprocessed = list(source_docs_path.rglob("*.md")) + list(source_docs_path.rglob("*.pdf"))
        logger.info(f"Found {len(unprocessed)} documents to process")

    model = ChatOpenAI(
        model=settings.model.name,
        temperature=settings.model.temperature,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

    pipeline = create_ingestion_pipeline(
        settings=settings,
        chroma_store=chroma_store,
        file_tracker=file_tracker,
        model=model,
    )

    success_count = 0
    error_count = 0

    for file_path in unprocessed:
        logger.info(f"Processing: {file_path}")

        initial_state = {
            "file_path": str(file_path),
            "document": None,
            "chunks": [],
            "privacy_results": [],
            "metadatas": [],
            "errors": [],
            "settings": settings,
            "chroma_store": chroma_store,
            "file_tracker": file_tracker,
            "Model_client": model,
        }

        result = pipeline.invoke(initial_state)

        if result["errors"]:
            logger.warning(f"Skipped {file_path}: {result['errors']}")
            error_count += 1
        else:
            success_count += 1

    stats = chroma_store.get_collection_stats()

    logger.info(f"✅ Ingestion complete: {success_count} succeeded, {error_count} skipped")
    logger.info(f"📊 Knowledge base ready with {stats['total_chunks']} chunks")


if __name__ == "__main__":
    main()
