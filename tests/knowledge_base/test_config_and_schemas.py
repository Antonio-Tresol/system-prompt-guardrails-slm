"""Unit tests for knowledge base configuration and schemas."""

from pathlib import Path

import pytest
import yaml

from knowledge_base.config.settings import Settings
from knowledge_base.schemas.chunk_metadata import ChunkMetadata


class TestSettings:
    """Test settings loading and validation."""

    def test_load_from_yaml_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful loading of settings from YAML."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://test.com")

        config_dir = tmp_path / "config"
        config_dir.mkdir()

        config_data = {
            "paths": {
                "source_documents": "./data",
                "vector_db": "./db",
                "pdf_private_config": str(config_dir / "private.yaml"),
                "file_tracker": "./tracker.json",
            },
            "embeddings": {"model": "test-model"},
            "llm": {"model": "test-llm", "temperature": 0.0},
            "chunking": {"max_chunk_size": 1000, "min_chunk_size": 100},
            "logging": {"level": "INFO"},
        }

        config_file = config_dir / "config.yaml"
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        private_config = config_dir / "private.yaml"
        with private_config.open("w") as f:
            yaml.dump(
                {
                    "cookbook_private_keywords": ["Secret"],
                    "paper_private_keywords": ["Methodology"],
                },
                f,
            )

        settings = Settings.load_from_yaml(str(config_file))

        assert settings.paths.source_documents == "./data"
        assert settings.embeddings.model == "test-model"
        assert settings.llm.temperature == 0.0
        assert "Secret" in settings.private_keywords
        assert "Methodology" in settings.private_keywords
        assert settings.openrouter_api_key == "test-key"

    def test_load_private_keywords_missing_file(self) -> None:
        """Test loading private keywords when file doesn't exist."""
        keywords = Settings._load_private_keywords("nonexistent.yaml")
        assert keywords == []


class TestChunkMetadata:
    """Test chunk metadata schema."""

    def test_chunk_metadata_creation(self) -> None:
        """Test creating chunk metadata with all fields."""
        metadata = ChunkMetadata(
            document_title="Test Doc",
            section="Section 1",
            subsection="Subsection 1.1",
            has_private_info=True,
            privacy_level="mixed",
            num_tokens=100,
            num_words=50,
            char_start=0,
            char_end=500,
            chunk_index=0,
            source_file="test.md",
            page_number=1,
            heading_level=2,
        )

        assert metadata.document_title == "Test Doc"
        assert metadata.privacy_level == "mixed"
        assert metadata.num_tokens == 100

    def test_chunk_metadata_to_dict(self) -> None:
        """Test converting metadata to dictionary."""
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

        metadata_dict = metadata.to_dict()

        assert isinstance(metadata_dict, dict)
        assert metadata_dict["document_title"] == "Test"
        assert metadata_dict["privacy_level"] == "public"
        assert metadata_dict["subsection"] is None
