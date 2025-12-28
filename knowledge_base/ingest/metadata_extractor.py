import tiktoken

from knowledge_base.ingest.chunkers import Chunk
from knowledge_base.ingest.privacy_detector import PrivacyResult
from knowledge_base.schemas.chunk_metadata import ChunkMetadata


def extract_metadata(
    *,
    chunk: Chunk,
    privacy_result: PrivacyResult,
    chunk_idx: int,
    document_title: str,
    source_file: str,
    start_page: int | None = None,
    end_page: int | None = None,
    page_number: int | None = None,
) -> ChunkMetadata:
    """Extract full metadata from a chunk.

    Args:
        chunk: The chunk to extract metadata from.
        privacy_result: Privacy detection results.
        chunk_idx: Index of the chunk in the document.
        document_title: Title of the document.
        source_file: Path to the source file.
        start_page: Starting page number.
        end_page: Ending page number.
        page_number: Legacy page number.

    Returns:
        Complete chunk metadata.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(chunk.text)
    num_tokens = len(tokens)

    words = chunk.text.split()
    num_words = len(words)

    # Extract section from headings (most reliable)
    headings = chunk.meta.get("headings", [])
    if headings and isinstance(headings, list):
        # Use the last heading as the most specific section
        section = headings[-1] if headings else "Unknown Section"
    else:
        section = "Unknown Section"

    # Subsection can be extracted if there are multiple headings
    subsection = None
    if len(headings) > 2:
        subsection = headings[-2]

    # Heading level based on number of headings (depth)
    heading_level = len(headings) if headings else None

    return ChunkMetadata(
        document_title=document_title,
        section=str(section),
        subsection=str(subsection) if subsection else None,
        has_private_info=privacy_result.has_private_info,
        privacy_level=privacy_result.privacy_level,
        num_tokens=num_tokens,
        num_words=num_words,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        start_page=start_page,
        end_page=end_page,
        chunk_index=chunk_idx,
        source_file=source_file,
        page_number=page_number,
        heading_level=heading_level,
    )
