from typing import Literal

from pydantic import BaseModel, Field


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
    start_line: int | None = Field(default=None, description="Starting line number in source")
    end_line: int | None = Field(default=None, description="Ending line number in source")
    start_page: int | None = Field(default=None, description="Starting page number (for PDFs)")
    end_page: int | None = Field(default=None, description="Ending page number (for PDFs)")
    chunk_index: int
    source_file: str
    page_number: int | None
    heading_level: int | None

    def to_dict(self) -> dict[str, str | int | bool | None]:
        """Convert metadata to dictionary for ChromaDB storage."""
        return self.model_dump()
