"""Integration tests for privacy detector using real model calls.

These tests validate the detect_privacy function with actual LLM API calls
to ensure structured output works correctly and handles parsing errors.
"""

import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from knowledge_base.ingest.chunkers import Chunk
from knowledge_base.ingest.privacy_detector import PrivacyResult, detect_privacy

# Load environment variables from .env file
load_dotenv()


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="Requires OPENROUTER_API_KEY env var",
)
class TestPrivacyDetectorIntegration:
    """Integration tests for privacy detection with real model calls."""

    @pytest.fixture
    def model(self) -> ChatOpenAI:
        """Create a ChatOpenAI model for testing."""
        return ChatOpenAI(
            model="openai/gpt-oss-120b",
            temperature=0.0,
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    @pytest.fixture
    def private_keywords(self) -> list[str]:
        """Common privacy keywords for testing."""
        return ["Internal", "Restricted", "salary", "confidential", "secret"]

    def test_detect_public_content(
        self,
        *,
        model: ChatOpenAI,
        private_keywords: list[str],
    ) -> None:
        """Test detecting public content returns public privacy level.

        Validates that clearly public content is correctly classified
        as having no private information.
        """
        # Arrange
        chunk = Chunk(
            text="This is a delicious recipe for chocolate chip cookies. "
            "Preheat the oven to 350°F and mix the ingredients thoroughly.",
            char_start=0,
            char_end=120,
            meta={"doc_items": [{"label": "Recipe"}]},
        )

        # Act
        result = detect_privacy(
            chunk=chunk,
            keywords=private_keywords,
            model=model,
        )

        # Assert
        assert isinstance(result, PrivacyResult)
        assert result.privacy_level == "public"
        assert result.has_private_info is False
        assert len(result.reasoning) > 0

    def test_detect_private_content(
        self,
        *,
        model: ChatOpenAI,
        private_keywords: list[str],
    ) -> None:
        """Test detecting private content returns private/mixed privacy level.

        Validates that content with privacy markers is correctly classified.
        """
        # Arrange
        chunk = Chunk(
            text="Restricted Section: Internal salary data for employees. "
            "Manager salary: $85,000. This is confidential information.",
            char_start=0,
            char_end=120,
            meta={"doc_items": [{"label": "HR Section"}]},
        )

        # Act
        result = detect_privacy(
            chunk=chunk,
            keywords=private_keywords,
            model=model,
        )

        # Assert
        assert isinstance(result, PrivacyResult)
        assert result.privacy_level in ["private", "mixed"]
        assert result.has_private_info is True
        assert len(result.reasoning) > 0

    def test_detect_mixed_content(
        self,
        *,
        model: ChatOpenAI,
        private_keywords: list[str],
    ) -> None:
        """Test detecting mixed content with both public and private sections.

        Validates that content containing both public and private information
        is correctly classified as mixed.
        """
        # Arrange
        chunk = Chunk(
            text="Welcome to our company cookbook! Here's a public recipe for pasta. "
            "Internal note: Staff cost for ingredients is $50 per serving.",
            char_start=0,
            char_end=150,
            meta={"doc_items": [{"label": "Mixed Section"}]},
        )

        # Act
        result = detect_privacy(
            chunk=chunk,
            keywords=private_keywords,
            model=model,
        )

        # Assert
        assert isinstance(result, PrivacyResult)
        assert result.privacy_level in ["mixed", "private"]
        assert len(result.reasoning) > 0

    def test_returns_valid_result_on_empty_keywords(
        self,
        *,
        model: ChatOpenAI,
    ) -> None:
        """Test that detection works even with empty keywords list.

        Validates that the function handles edge case of no keywords gracefully.
        """
        # Arrange
        chunk = Chunk(
            text="This is regular public content without any private markers.",
            char_start=0,
            char_end=60,
            meta={"doc_items": [{"label": "Public"}]},
        )

        # Act
        result = detect_privacy(
            chunk=chunk,
            keywords=[],
            model=model,
        )

        # Assert
        assert isinstance(result, PrivacyResult)
        assert result.privacy_level in ["public", "mixed", "private"]

    def test_handles_long_chunk_content(
        self,
        *,
        model: ChatOpenAI,
        private_keywords: list[str],
    ) -> None:
        """Test detection handles longer chunks correctly.

        Validates that the function works with realistic chunk sizes.
        """
        # Arrange
        long_content = (
            "This is a detailed section about cooking techniques. " * 20
            + "The content is entirely public and meant for general audiences."
        )
        chunk = Chunk(
            text=long_content,
            char_start=0,
            char_end=len(long_content),
            meta={"doc_items": [{"label": "Cooking Techniques"}]},
        )

        # Act
        result = detect_privacy(
            chunk=chunk,
            keywords=private_keywords,
            model=model,
        )

        # Assert
        assert isinstance(result, PrivacyResult)
        assert result.privacy_level in ["public", "mixed", "private"]
        assert len(result.reasoning) > 0

    def test_structured_output_returns_correct_schema(
        self,
        *,
        model: ChatOpenAI,
        private_keywords: list[str],
    ) -> None:
        """Test that structured output matches PrivacyResult schema.

        Validates that all required fields are present and correctly typed.
        """
        # Arrange
        chunk = Chunk(
            text="A simple test chunk for schema validation.",
            char_start=0,
            char_end=45,
            meta={"doc_items": [{"label": "Test"}]},
        )

        # Act
        result = detect_privacy(
            chunk=chunk,
            keywords=private_keywords,
            model=model,
        )

        # Assert: Validate schema
        assert hasattr(result, "has_private_info")
        assert hasattr(result, "privacy_level")
        assert hasattr(result, "reasoning")
        assert isinstance(result.has_private_info, bool)
        assert result.privacy_level in ["public", "mixed", "private"]
        assert isinstance(result.reasoning, str)
