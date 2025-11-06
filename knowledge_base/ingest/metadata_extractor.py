import tiktoken

from knowledge_base.ingest.chunkers import Chunk
from knowledge_base.ingest.privacy_detector import PrivacyResult
from knowledge_base.schemas.chunk_metadata import ChunkMetadata


def extract_metadata(
    chunk: Chunk,
    privacy_result: PrivacyResult,
    chunk_idx: int,
    document_title: str,
    source_file: str,
) -> ChunkMetadata:
    """Extract full metadata from a chunk.

    Args:
        chunk: The chunk to extract metadata from.
        privacy_result: Privacy detection results.
        chunk_idx: Index of the chunk in the document.
        document_title: Title of the document.
        source_file: Path to the source file.

    Returns:
        Complete chunk metadata.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(chunk.text)
    num_tokens = len(tokens)

    words = chunk.text.split()
    num_words = len(words)

    section = chunk.meta.get("doc_items", [{}])[0].get("label", "Unknown Section")
    if isinstance(section, list):
        section = section[0] if section else "Unknown Section"

    subsection = None
    if len(chunk.meta.get("doc_items", [])) > 1:
        subsection = chunk.meta.get("doc_items", [{}])[1].get("label")

    page_number = chunk.meta.get("origin", {}).get("page_number")
    heading_level = (
        chunk.meta.get("headings", [{}])[0].get("level") if chunk.meta.get("headings") else None
    )

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
        chunk_index=chunk_idx,
        source_file=source_file,
        page_number=page_number,
        heading_level=heading_level,
    )
