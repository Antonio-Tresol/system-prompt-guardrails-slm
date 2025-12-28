from typing import Any

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from pydantic import BaseModel


class Chunk(BaseModel):
    """Represents a chunk of text from a document."""

    text: str
    char_start: int
    char_end: int
    start_line: int | None = None
    end_line: int | None = None
    start_page: int | None = None
    end_page: int | None = None
    meta: dict[str, Any] = {}


def chunk_document(
    *,
    doc: Any,  # noqa: ANN401
    max_chunk_size: int,
    min_chunk_size: int,
    raw_text: str | None = None,
) -> list[Chunk]:
    """Chunk a document using Docling's native chunking API.

    Args:
        doc: Docling document object.
        max_chunk_size: Maximum chunk size in tokens.
        min_chunk_size: Minimum chunk size in tokens.
        raw_text: Optional raw source text for precise line mapping.

    Returns:
        List of chunks with text and positional metadata.
    """
    # Use tiktoken with OpenAI tokenizer for consistent tokenization
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=max_chunk_size,
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )

    # Use raw_text if provided, otherwise export from doc
    if raw_text is not None:
        full_text = raw_text
    else:
        try:
            full_text = doc.export_to_markdown()
        except AttributeError:
            full_text = getattr(doc, "text", "")

    # Pre-calculate line offsets
    lines = full_text.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines[:-1]:
        line_offsets.append(line_offsets[-1] + len(line))

    # We need a way to map normalized indices back to original indices
    # However, simpler: search for significant parts of chunk_text

    current_search_pos = 0
    chunks: list[Chunk] = []
    chunk_iter = chunker.chunk(dl_doc=doc)

    for _chunk_idx, chunk in enumerate(chunk_iter):
        chunk_text = chunk.text
        if len(chunk_text.split()) < min_chunk_size // 4:
            continue

        # Try exact find first
        start_char = full_text.find(chunk_text, current_search_pos)

        if start_char == -1:
            # Fallback 1: Try finding a normalized version of the chunk
            # This is slow, so we only do it if exact fails.
            # Faster fallback: search for a few representative lines from the chunk
            chunk_lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
            if chunk_lines:
                # Try middle line as it is usually most unique/stable
                probe = chunk_lines[len(chunk_lines) // 2]
                if len(probe) > 10:
                    start_char = full_text.find(probe, current_search_pos)
                    if start_char != -1:
                        # Success, but we need to adjust back to approximate start of chunk
                        # Actually, let us just keep this start_char as a "best guess" for line mapping
                        pass

        if start_char != -1:
            end_char = start_char + len(chunk_text)
            current_search_pos = start_char + 1

            # Find line numbers
            import bisect

            idx_start = bisect.bisect_right(line_offsets, start_char) - 1
            start_line = idx_start + 1  # 1-indexed

            target_end = max(0, end_char - 1)
            idx_end = bisect.bisect_right(line_offsets, target_end) - 1
            end_line = idx_end + 1  # 1-indexed

        else:
            start_char = 0
            end_char = len(chunk_text)
            start_line = None
            end_line = None

        # Extract Page Numbers from Docling Metadata
        # Docling v2 stores provenance in chunk.meta.doc_items
        start_page = None
        end_page = None

        try:
            # Check for doc_items in meta
            doc_items = getattr(chunk.meta, "doc_items", [])
            page_nos = []
            for item in doc_items:
                if hasattr(item, "prov") and item.prov:
                    # item.prov is a list of ProvenanceItem in v2
                    for p in item.prov:
                        if hasattr(p, "page_no") and p.page_no:
                            page_nos.append(p.page_no)

            if page_nos:
                start_page = min(page_nos)
                end_page = max(page_nos)
        except Exception:
            # Fallback to legacy or empty
            pass

        meta_dict = chunk.meta.model_dump() if hasattr(chunk.meta, "model_dump") else {}

        chunks.append(
            Chunk(
                text=chunk_text,
                char_start=start_char,
                char_end=end_char,
                start_line=start_line,
                end_line=end_line,
                start_page=start_page,
                end_page=end_page,
                meta=meta_dict,
            ),
        )

    return chunks
