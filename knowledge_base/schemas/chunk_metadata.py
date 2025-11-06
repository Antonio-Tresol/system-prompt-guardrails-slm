from typing import Literal

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Rich metadata for each chunk in the vector database."""

    document_title: str
    section: str
    subsection: str | None
    has_private_info: bool
    privacy_level: Literal["public", "mixed", "private"]
    num_tokens: int
    num_words: int
    char_start: int
    char_end: int
    chunk_index: int
    source_file: str
    page_number: int | None
    heading_level: int | None

    def to_dict(self) -> dict[str, str | int | bool | None]:
        """Convert metadata to dictionary for ChromaDB storage."""
        return self.model_dump()
