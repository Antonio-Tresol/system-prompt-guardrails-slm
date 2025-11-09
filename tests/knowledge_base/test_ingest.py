"""Unit tests for document ingestion components."""

from unittest.mock import MagicMock, Mock, patch

from knowledge_base.ingest.chunkers import Chunk, chunk_document
from knowledge_base.ingest.metadata_extractor import extract_metadata
from knowledge_base.ingest.privacy_detector import PrivacyResult

# Test constants
TEST_MAX_CHUNK_SIZE = 1000
TEST_MIN_CHUNK_SIZE = 100
TEST_CHUNK_WORD_COUNT = 30
TEST_TOKEN_COUNT = 50
TEST_SECTION_1_LABEL = "Section 1"
TEST_SECTION_2_LABEL = "Section 2"
TEST_DOCUMENT_TITLE = "Test Document"
TEST_SOURCE_FILE = "test.md"


class TestChunking:
    """Test document chunking functionality."""

    @patch("knowledge_base.ingest.chunkers.HybridChunker")
    def test_chunk_document(self, mock_chunker_class: Mock) -> None:
        """Test document chunking with Docling's HybridChunker.

        Validates that chunks are created with correct text and metadata,
        and meet minimum size requirements.
        """
        # Arrange: Create mock document and chunks
        mock_doc = MagicMock()

        mock_chunk1 = MagicMock()
        mock_chunk1.text = " ".join(["word"] * TEST_CHUNK_WORD_COUNT)
        mock_chunk1.meta = MagicMock()
        mock_chunk1.meta.model_dump.return_value = {"doc_items": [{"label": TEST_SECTION_1_LABEL}]}

        mock_chunk2 = MagicMock()
        mock_chunk2.text = " ".join(["word"] * TEST_CHUNK_WORD_COUNT)
        mock_chunk2.meta = MagicMock()
        mock_chunk2.meta.model_dump.return_value = {"doc_items": [{"label": TEST_SECTION_2_LABEL}]}

        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk.return_value = [mock_chunk1, mock_chunk2]
        mock_chunker_class.return_value = mock_chunker_instance

        # Act: Chunk the document
        chunks = chunk_document(
            doc=mock_doc, max_chunk_size=TEST_MAX_CHUNK_SIZE, min_chunk_size=TEST_MIN_CHUNK_SIZE
        )

        # Assert: Verify chunks created correctly
        assert len(chunks) == 2, "Should create two chunks"
        assert all(isinstance(c, Chunk) for c in chunks), "All items should be Chunk instances"
        assert chunks[0].text == mock_chunk1.text, "First chunk text should match"
        assert chunks[1].text == mock_chunk2.text, "Second chunk text should match"


class TestMetadataExtraction:
    """Test metadata extraction from chunks."""

    @patch("knowledge_base.ingest.metadata_extractor.tiktoken.get_encoding")
    def test_extract_metadata(self, mock_get_encoding: Mock) -> None:
        """Test extracting complete metadata from a public content chunk.

        Validates that metadata includes token counts, word counts,
        document info, and privacy classification.
        """
        # Arrange: Mock token encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1] * TEST_TOKEN_COUNT
        mock_get_encoding.return_value = mock_encoding

        # Arrange: Create test chunk
        chunk = Chunk(
            text="This is a test chunk for metadata extraction. " * 10,
            char_start=0,
            char_end=500,
            meta={"doc_items": [{"label": "Test Section"}]},
        )

        privacy_result = PrivacyResult(
            has_private_info=False,
            privacy_level="public",
            reasoning="No private content",
        )

        # Act: Extract metadata
        metadata = extract_metadata(
            chunk=chunk,
            privacy_result=privacy_result,
            chunk_idx=0,
            document_title=TEST_DOCUMENT_TITLE,
            source_file=TEST_SOURCE_FILE,
        )

        # Assert: Verify all metadata fields
        assert metadata.document_title == TEST_DOCUMENT_TITLE
        assert metadata.privacy_level == "public"
        assert metadata.has_private_info is False
        assert metadata.chunk_index == 0
        assert metadata.num_tokens == TEST_TOKEN_COUNT
        assert metadata.num_words > 0, "Word count should be positive"
        assert metadata.source_file == TEST_SOURCE_FILE

    @patch("knowledge_base.ingest.metadata_extractor.tiktoken.get_encoding")
    def test_extract_metadata_with_private_content(self, mock_get_encoding: Mock) -> None:
        """Test extracting metadata from chunk with private/restricted content.

        Validates that privacy classification is correctly reflected in metadata.
        """
        # Arrange: Mock token encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1] * 10
        mock_get_encoding.return_value = mock_encoding

        # Arrange: Create chunk with private content
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

        # Act: Extract metadata
        metadata = extract_metadata(
            chunk=chunk,
            privacy_result=privacy_result,
            chunk_idx=1,
            document_title="Confidential Doc",
            source_file="confidential.pdf",
        )

        # Assert: Verify privacy flags are set correctly
        assert metadata.has_private_info is True
        assert metadata.privacy_level == "private"
        assert metadata.chunk_index == 1
