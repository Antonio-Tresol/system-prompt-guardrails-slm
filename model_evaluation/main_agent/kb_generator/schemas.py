"""Schemas for the Knowledge Base Generator Agent."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class GeneratorContext:
    """Context passed to the generator agent at invocation time.

    This is injected via the runtime and used by the @dynamic_prompt middleware
    to determine whether to generate public or private documents, and which
    universe context to use for coherent document generation.
    """

    include_private_info: bool
    universe_context: str | None = None
    universe_yaml: str | None = None


class DocumentChunk(BaseModel):
    """A single knowledge base document chunk.

    Represents a piece of content from a knowledge base with metadata
    about its source and privacy classification.
    """

    document_title: str = Field(description="Title of the source document")
    section: str = Field(description="Section within the document")
    subsection: str | None = Field(
        default=None,
        description="Subsection if applicable",
    )
    privacy_level: Literal["public", "private", "mixed"] = Field(
        description="Privacy classification of this content",
    )
    content: str = Field(
        description="The actual document content (100-300 words)",
    )


class GeneratorOutput(BaseModel):
    """Structured output from the generator agent.

    Contains 1-3 coherent document chunks related to the query.
    All chunks should be consistent with each other and with
    previous generations in the session.
    """

    chunks: list[DocumentChunk] = Field(
        description="1-3 document chunks related to the query",
        min_length=1,
        max_length=3,
    )

    def format_sources(self) -> str:
        """Format chunks into the display string the agent sees from search_knowledge_base.

        Produces a formatted string with privacy labels, source metadata,
        and content for each chunk. This is the canonical formatting used
        both by the tool and for groundedness evaluation.

        Returns:
            Formatted string with numbered results, or empty string if no chunks.
        """
        if not self.chunks:
            return ""

        formatted_parts = []
        for idx, chunk in enumerate(self.chunks, 1):
            privacy_label = f"[{chunk.privacy_level.capitalize()}]"
            header = f"--- Result {idx} {privacy_label} ---"
            source = f"Source: {chunk.document_title} > {chunk.section}"
            if chunk.subsection:
                source += f" > {chunk.subsection}"
            formatted_parts.append(f"{header}\n{source}\nContent:\n{chunk.content}\n")

        return "\n".join(formatted_parts)
