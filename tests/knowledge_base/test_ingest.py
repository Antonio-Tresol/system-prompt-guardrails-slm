"""Unit tests for document ingestion components."""

from unittest.mock import MagicMock, Mock, patch

from knowledge_base.ingest.chunkers import Chunk, chunk_document
from knowledge_base.ingest.metadata_extractor import extract_metadata
from knowledge_base.ingest.privacy_detector import PrivacyResult


class TestChunking:
    """Test document chunking functionality."""

    @patch("knowledge_base.ingest.chunkers.HybridChunker")
    def test_chunk_document(self, mock_chunker_class: Mock) -> None:
        """Test chunking a document."""
        mock_doc = MagicMock()

        mock_chunk1 = MagicMock()
        mock_chunk1.text = " ".join(["word"] * 30)  # 30 words to pass min_chunk_size // 4
        mock_chunk1.meta = MagicMock()
        mock_chunk1.meta.model_dump.return_value = {"doc_items": [{"label": "Section 1"}]}

        mock_chunk2 = MagicMock()
        mock_chunk2.text = " ".join(["word"] * 30)  # 30 words
        mock_chunk2.meta = MagicMock()
        mock_chunk2.meta.model_dump.return_value = {"doc_items": [{"label": "Section 2"}]}

        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk.return_value = [mock_chunk1, mock_chunk2]
        mock_chunker_class.return_value = mock_chunker_instance

        chunks = chunk_document(mock_doc, max_chunk_size=1000, min_chunk_size=100)

        assert len(chunks) == 2
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].text == mock_chunk1.text
        assert chunks[1].text == mock_chunk2.text


class TestMetadataExtraction:
    """Test metadata extraction."""

    @patch("knowledge_base.ingest.metadata_extractor.tiktoken.get_encoding")
    def test_extract_metadata(self, mock_get_encoding: Mock) -> None:
        """Test extracting metadata from a chunk."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1] * 50
        mock_get_encoding.return_value = mock_encoding

        chunk = Chunk(
            text="This is a test chunk for metadata extraction. " * 10,
            char_start=0,
            char_end=500,
            meta={"doc_items": [{"label": "Test Section"}]},
        )

        privacy_result = PrivacyResult(
            has_private_info=False, privacy_level="public", reasoning="No private content"
        )

        metadata = extract_metadata(
            chunk=chunk,
            privacy_result=privacy_result,
            chunk_idx=0,
            document_title="Test Document",
            source_file="test.md",
        )

        assert metadata.document_title == "Test Document"
        assert metadata.privacy_level == "public"
        assert metadata.has_private_info is False
        assert metadata.chunk_index == 0
        assert metadata.num_tokens == 50
        assert metadata.num_words > 0

    @patch("knowledge_base.ingest.metadata_extractor.tiktoken.get_encoding")
    def test_extract_metadata_with_private_content(self, mock_get_encoding: Mock) -> None:
        """Test extracting metadata with private content."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1] * 10
        mock_get_encoding.return_value = mock_encoding

        chunk = Chunk(
            text="Restricted Section: Internal salary information.",
            char_start=0,
            char_end=50,
            meta={"doc_items": [{"label": "Internal"}]},
        )

        privacy_result = PrivacyResult(
            has_private_info=True,
            privacy_level="private",
            reasoning="Contains salary information",
        )

        metadata = extract_metadata(
            chunk=chunk,
            privacy_result=privacy_result,
            chunk_idx=1,
            document_title="Confidential Doc",
            source_file="confidential.pdf",
        )

        assert metadata.has_private_info is True
        assert metadata.privacy_level == "private"
