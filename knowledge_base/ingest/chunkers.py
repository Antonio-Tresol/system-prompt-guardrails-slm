from typing import Any

from docling.chunking import HybridChunker
from pydantic import BaseModel


class Chunk(BaseModel):
    """Represents a chunk of text from a document."""

    text: str
    char_start: int
    char_end: int
    meta: dict[str, Any] = {}


def chunk_document(
    *,
    doc: Any,  # noqa: ANN401
    max_chunk_size: int,
    min_chunk_size: int,
) -> list[Chunk]:
    """Chunk a document using Docling's native chunking API.

    Args:
        doc: Docling document object.
        max_chunk_size: Maximum chunk size in tokens.
        min_chunk_size: Minimum chunk size in tokens.

    Returns:
        List of chunks with text and positional metadata.
    """
    chunker = HybridChunker(
        max_tokens=max_chunk_size,
    )

    chunks: list[Chunk] = []
    chunk_iter = chunker.chunk(dl_doc=doc)

    for _chunk_idx, chunk in enumerate(chunk_iter):
        chunk_text = chunk.text
        if len(chunk_text.split()) < min_chunk_size // 4:
            continue

        meta_dict = chunk.meta.model_dump() if hasattr(chunk.meta, "model_dump") else {}

        chunks.append(
            Chunk(
                text=chunk_text,
                char_start=0,
                char_end=len(chunk_text),
                meta=meta_dict,
            )
        )

    return chunks
