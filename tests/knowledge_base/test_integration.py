"""End-to-end integration tests for knowledge base pipeline."""

import os
from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI

from knowledge_base.config.settings import Settings
from knowledge_base.ingest.pipeline import create_ingestion_pipeline
from knowledge_base.utils.file_tracker import FileTracker
from knowledge_base.vectordb.chroma_store import ChromaStore


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="Requires OPENROUTER_API_KEY env var"
)
class TestEndToEndPipeline:
    """End-to-end integration tests with real API calls."""

    def test_full_pipeline_with_markdown(self, tmp_path: Path) -> None:
        """Test complete pipeline with a real markdown document."""
        # Create test markdown document
        test_md = tmp_path / "test_doc.md"
        test_md.write_text("""# Test Document

## Public Section

This is a public section with general information about cooking.

## Restricted Section

Internal Notes: This section contains private salary information.
Staff salary: $50,000
""")

        # Create necessary directories
        vector_db_dir = tmp_path / "vector_db"
        tracker_file = tmp_path / "tracker.json"

        # Initialize components
        file_tracker = FileTracker(str(tracker_file))

        chroma_store = ChromaStore(
            persist_directory=str(vector_db_dir),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", "openai/text-embedding-3-small"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

        llm_client = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            temperature=0.0,
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

        # Create minimal settings for testing
        from knowledge_base.config.settings import (
            ChunkingConfig,
            EmbeddingsConfig,
            LLMConfig,
            LoggingConfig,
            PathsConfig,
        )

        settings = Settings(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            paths=PathsConfig(
                source_documents=str(tmp_path),
                vector_db=str(vector_db_dir),
                pdf_private_config="",
                file_tracker=str(tracker_file),
            ),
            embeddings=EmbeddingsConfig(model="openai/text-embedding-3-small"),
            llm=LLMConfig(model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"), temperature=0.0),
            chunking=ChunkingConfig(max_chunk_size=1000, min_chunk_size=20),
            logging=LoggingConfig(level="INFO"),
            private_keywords=["Internal", "Restricted", "salary"],
        )

        # Create pipeline
        pipeline = create_ingestion_pipeline(settings, chroma_store, file_tracker, llm_client)

        # Run pipeline
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
            "llm_client": llm_client,
        }

        result = pipeline.invoke(initial_state)

        # Assertions
        assert not result["errors"], f"Pipeline had errors: {result['errors']}"
        assert len(result["chunks"]) > 0, "No chunks were created"
        assert len(result["privacy_results"]) > 0, "No privacy results"
        assert len(result["metadatas"]) > 0, "No metadata extracted"

        # Verify privacy detection worked
        privacy_levels = [pr.privacy_level for pr in result["privacy_results"]]
        assert any(level in ["mixed", "private"] for level in privacy_levels), (
            "Privacy detection should have found private content"
        )

        # Verify metadata
        for metadata in result["metadatas"]:
            assert metadata.num_tokens > 0
            assert metadata.num_words > 0
            assert metadata.document_title == "test doc"

        # Query the vector store
        query_results = chroma_store.vector_store.similarity_search(
            "public cooking information", k=5
        )

        assert len(query_results) > 0, "Vector store should return results"

        # Verify we can filter by privacy level
        stats = chroma_store.get_collection_stats()
        assert stats["total_chunks"] > 0

        print("\n✅ Pipeline test passed!")
        print(f"   - Processed {len(result['chunks'])} chunks")
        print(f"   - Privacy levels found: {set(privacy_levels)}")
        print(f"   - Total chunks in DB: {stats['total_chunks']}")
        print(f"   - Query returned {len(query_results)} results")
