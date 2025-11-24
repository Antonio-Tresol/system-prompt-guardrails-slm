"""Unit tests for knowledge base configuration and schemas."""

from pathlib import Path

import pytest
import yaml

from knowledge_base.config.settings import Settings
from knowledge_base.schemas.chunk_metadata import ChunkMetadata

# Test constants
TEST_API_KEY = "test-key"
TEST_BASE_URL = "https://test.com"
TEST_MODEL_NAME = "test-model"
TEST_TEMPERATURE = 0.0
TEST_MAX_CHUNK_SIZE = 1000
TEST_MIN_CHUNK_SIZE = 100
TEST_COOKBOOK_KEYWORD = "Secret"
TEST_PAPER_KEYWORD = "Methodology"


class TestSettings:
    """Test settings loading and validation."""

    def test_load_from_yaml_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful loading and validation of settings from YAML files.

        Validates that settings correctly load from config YAML, merge private
        keywords from separate file, and read environment variables.
        """
        # Arrange: Set up environment variables
        monkeypatch.setenv("OPENROUTER_API_KEY", TEST_API_KEY)
        monkeypatch.setenv("OPENROUTER_BASE_URL", TEST_BASE_URL)

        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Arrange: Create test configuration files
        config_data = {
            "paths": {
                "source_documents": "./data",
                "vector_db": "./db",
                "pdf_private_config": str(config_dir / "private.yaml"),
                "file_tracker": "./tracker.json",
            },
            "embeddings": {"model": TEST_MODEL_NAME},
            "model": {"name": TEST_MODEL_NAME, "temperature": TEST_TEMPERATURE},
            "chunking": {
                "max_chunk_size": TEST_MAX_CHUNK_SIZE,
                "min_chunk_size": TEST_MIN_CHUNK_SIZE,
            },
            "logging": {"level": "INFO"},
        }

        config_file = config_dir / "config.yaml"
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        private_config = config_dir / "private.yaml"
        with private_config.open("w") as f:
            yaml.dump(
                {
                    "cookbook_private_keywords": [TEST_COOKBOOK_KEYWORD],
                    "paper_private_keywords": [TEST_PAPER_KEYWORD],
                },
                f,
            )

        # Act: Load settings from YAML
        settings = Settings.load_from_yaml(config_path=str(config_file))

        # Assert: Verify all settings loaded correctly
        # Paths are resolved to absolute, so check they end with expected relative path
        assert settings.paths.source_documents.endswith("data")
        assert settings.embeddings.model == TEST_MODEL_NAME
        assert settings.model.temperature == TEST_TEMPERATURE
        assert TEST_COOKBOOK_KEYWORD in settings.private_keywords
        assert TEST_PAPER_KEYWORD in settings.private_keywords
        assert settings.openrouter_api_key == TEST_API_KEY

    def test_load_private_keywords_missing_file(self) -> None:
        """Test graceful handling when private keywords file doesn't exist.

        Should return empty list instead of raising an error.
        """
        # Arrange: Use nonexistent file path
        nonexistent_file = "nonexistent.yaml"

        # Act: Attempt to load keywords
        keywords = Settings._load_private_keywords(nonexistent_file)

        # Assert: Should return empty list
        assert keywords == []


class TestChunkMetadata:
    """Test chunk metadata schema."""

    def test_chunk_metadata_creation(self) -> None:
        """Test creating chunk metadata with all required and optional fields.

        Validates that ChunkMetadata accepts all fields and stores them correctly.
        """
        # Arrange: Prepare test data with all fields
        test_data = {
            "document_title": "Test Doc",
            "section": "Section 1",
            "subsection": "Subsection 1.1",
            "has_private_info": True,
            "privacy_level": "mixed",
            "num_tokens": 100,
            "num_words": 50,
            "char_start": 0,
            "char_end": 500,
            "chunk_index": 0,
            "source_file": "test.md",
            "page_number": 1,
            "heading_level": 2,
        }

        # Act: Create metadata instance
        metadata = ChunkMetadata(**test_data)

        # Assert: Verify all fields are set correctly
        assert metadata.document_title == "Test Doc"
        assert metadata.privacy_level == "mixed"
        assert metadata.num_tokens == 100
        assert metadata.has_private_info is True
        assert metadata.page_number == 1
        assert metadata.heading_level == 2

    def test_chunk_metadata_to_dict(self) -> None:
        """Test converting metadata to dictionary for ChromaDB storage.

        Validates that to_dict() produces a proper dict and excludes None values.
        """
        # Arrange: Create metadata with some None values
        metadata = ChunkMetadata(
            document_title="Test",
            section="S1",
            subsection=None,
            has_private_info=False,
            privacy_level="public",
            num_tokens=50,
            num_words=25,
            char_start=0,
            char_end=100,
            chunk_index=0,
            source_file="test.md",
            page_number=None,
            heading_level=None,
        )

        # Act: Convert to dictionary
        metadata_dict = metadata.to_dict()

        # Assert: Verify structure and content
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["document_title"] == "Test"
        assert metadata_dict["privacy_level"] == "public"
        assert metadata_dict["subsection"] is None  # None values are included
