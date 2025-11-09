"""End-to-end integration tests for knowledge base pipeline."""

import os
from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI

from knowledge_base.config.settings import (
    ChunkingConfig,
    EmbeddingsConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    Settings,
)
from knowledge_base.ingest.pipeline import create_ingestion_pipeline
from knowledge_base.utils.file_tracker import FileTracker
from knowledge_base.vectordb.chroma_store import ChromaStore

# Test constants
TEST_EMBEDDINGS_MODEL = "openai/text-embedding-3-small"
TEST_LLM_MODEL = "openai/gpt-4o-mini"
TEST_TEMPERATURE = 0.0
TEST_MAX_CHUNK_SIZE = 1000
TEST_MIN_CHUNK_SIZE = 20
TEST_PRIVATE_KEYWORDS = ["Internal", "Restricted", "salary"]

TEST_MARKDOWN_CONTENT = """# Test Document

## Public Section

This is a public section with general information about cooking.

## Restricted Section

Internal Notes: This section contains private salary information.
Staff salary: $50,000
"""


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="Requires OPENROUTER_API_KEY env var"
)
class TestEndToEndPipeline:
    """End-to-end integration tests with real API calls."""

    def test_full_pipeline_with_markdown(self, tmp_path: Path) -> None:
        """Test complete pipeline processes markdown and detects privacy correctly.

        This test validates:
        - Document loading and chunking
        - Privacy detection via LLM
        - Metadata extraction
        - Vector store insertion
        - Query functionality
        """
        # Arrange: Set up test document and directories
        test_md = tmp_path / "test_doc.md"
        test_md.write_text(TEST_MARKDOWN_CONTENT)

        vector_db_dir = tmp_path / "vector_db"
        tracker_file = tmp_path / "tracker.json"

        # Arrange: Initialize components
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        file_tracker = FileTracker(str(tracker_file))

        chroma_store = ChromaStore(
            persist_directory=str(vector_db_dir),
            embeddings_model=TEST_EMBEDDINGS_MODEL,
            openrouter_api_key=api_key,
            openrouter_base_url=base_url,
        )

        model = ChatOpenAI(
            model=TEST_LLM_MODEL,
            temperature=TEST_TEMPERATURE,
            api_key=api_key,
            base_url=base_url,
        )

        settings = Settings(
            openrouter_api_key=api_key,
            openrouter_base_url=base_url,
            paths=PathsConfig(
                source_documents=str(tmp_path),
                vector_db=str(vector_db_dir),
                pdf_private_config="",
                file_tracker=str(tracker_file),
            ),
            embeddings=EmbeddingsConfig(model=TEST_EMBEDDINGS_MODEL),
            model=ModelConfig(name=TEST_LLM_MODEL, temperature=TEST_TEMPERATURE),
            chunking=ChunkingConfig(
                max_chunk_size=TEST_MAX_CHUNK_SIZE, min_chunk_size=TEST_MIN_CHUNK_SIZE
            ),
            logging=LoggingConfig(level="INFO"),
            private_keywords=TEST_PRIVATE_KEYWORDS,
        )

        # Arrange: Create pipeline with initial state
        pipeline = create_ingestion_pipeline(
            settings=settings, chroma_store=chroma_store, file_tracker=file_tracker, model=model
        )

        initial_state = {
            "file_path": str(test_md),
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

        # Act: Run the ingestion pipeline
        result = pipeline.invoke(initial_state)

        # Assert: Pipeline completed without errors
        assert not result["errors"], f"Pipeline had errors: {result['errors']}"

        # Assert: Chunks were created
        assert len(result["chunks"]) > 0, "No chunks were created"
        assert len(result["privacy_results"]) > 0, "No privacy results generated"
        assert len(result["metadatas"]) > 0, "No metadata extracted"

        # Assert: Privacy detection identified restricted content
        privacy_levels = [pr.privacy_level for pr in result["privacy_results"]]
        assert any(level in ["mixed", "private"] for level in privacy_levels), (
            "Privacy detection should have found private content"
        )

        # Assert: Metadata is properly populated
        for metadata in result["metadatas"]:
            assert metadata.num_tokens > 0, "Token count should be positive"
            assert metadata.num_words > 0, "Word count should be positive"
            assert metadata.document_title == "test doc", "Document title mismatch"

        # Assert: Vector store is queryable
        query_results = chroma_store.vector_store.similarity_search(
            "public cooking information", k=5
        )
        assert len(query_results) > 0, "Vector store should return query results"

        # Assert: Collection stats are valid
        stats = chroma_store.get_collection_stats()
        total_chunks = stats["total_chunks"]
        assert isinstance(total_chunks, int), "total_chunks should be an integer"
        assert total_chunks > 0, "Vector store should contain chunks"

        # Output test summary
        print("\n✅ Pipeline test passed!")
        print(f"   - Processed {len(result['chunks'])} chunks")
        print(f"   - Privacy levels found: {set(privacy_levels)}")
        print(f"   - Total chunks in DB: {total_chunks}")
        print(f"   - Query returned {len(query_results)} results")
