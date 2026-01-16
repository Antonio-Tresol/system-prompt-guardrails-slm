"""Main entry point for building the knowledge base.

This module orchestrates the ingestion pipeline, handling data preparation,
document processing, and vector database population.
"""

import shutil
from pathlib import Path

from langchain_openai import ChatOpenAI

from knowledge_base.config.settings import Settings
from knowledge_base.ingest.pipeline import create_ingestion_pipeline
from knowledge_base.utils.file_tracker import FileTracker
from knowledge_base.vectordb.chroma_store import ChromaStore
from utils.logging import logger, setup_logging

SYNTHETIC_DATA_SOURCE = Path("data_generation/synthetic_data")


def _prepare_data_sources(*, source_dirs: list[Path]) -> None:
    """Ensure data directories are populated with required files.

    Args:
        source_dirs: List of source directories from config.
    """
    for source_dir in source_dirs:
        source_dir.mkdir(parents=True, exist_ok=True)

        if "cookbooks" in source_dir.name:
            existing_mds = list(source_dir.glob("*.md"))
            if not existing_mds and SYNTHETIC_DATA_SOURCE.exists():
                source_mds = list(SYNTHETIC_DATA_SOURCE.glob("*.md"))
                if source_mds:
                    logger.info(f"Copying {len(source_mds)} cookbooks from {SYNTHETIC_DATA_SOURCE}")
                    for md_file in source_mds:
                        shutil.copy2(md_file, source_dir / md_file.name)
                    logger.info(f"✅ Copied cookbooks to {source_dir}")
                else:
                    logger.warning(
                        f"No cookbooks found in {SYNTHETIC_DATA_SOURCE}. "
                        "Run 'uv run generate_data' to create them."
                    )

        elif "papers" in source_dir.name:
            existing_pdfs = list(source_dir.glob("*.pdf"))
            if not existing_pdfs:
                logger.info(f"No papers found in {source_dir}. Downloading papers...")
                from data_generation.papers.download_papers import (
                    ARXIV_URLS,
                    download_all_papers,
                )

                try:
                    download_all_papers(ARXIV_URLS, source_dir)
                except Exception as e:
                    logger.error(f"Failed to download papers: {e}")


def main() -> None:
    """Build and populate vector database."""
    settings = Settings.load_from_yaml()
    setup_logging(level=settings.logging.level, log_file="knowledge_base.log")

    logger.info("Starting knowledge base ingestion")

    source_dirs = [Path(p) for p in settings.paths.source_documents]

    _prepare_data_sources(source_dirs=source_dirs)

    file_tracker = FileTracker(settings.paths.file_tracker)

    chroma_store = ChromaStore(
        persist_directory=settings.paths.vector_db,
        embeddings_model=settings.embeddings.model,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
    )

    unprocessed = []
    if chroma_store.collection_exists():
        for source_path in source_dirs:
            if source_path.exists():
                unprocessed.extend(file_tracker.get_unprocessed_files(source_path))

        if not unprocessed:
            logger.info("✅ Vector DB is up-to-date")
            return
        logger.info(f"Found {len(unprocessed)} new/modified documents")
    else:
        logger.info("Creating new vector database")
        for source_path in source_dirs:
            if source_path.exists():
                unprocessed.extend(
                    list(source_path.rglob("*.md")) + list(source_path.rglob("*.pdf"))
                )
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
            "model": model,
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
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
