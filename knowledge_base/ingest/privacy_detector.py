import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from knowledge_base.ingest.chunkers import Chunk

logger = logging.getLogger(__name__)


class PrivacyResult(BaseModel):
    """Result of privacy detection."""

    has_private_info: bool = Field(
        description="True if the chunk contains any private/restricted information"
    )
    privacy_level: Literal["public", "mixed", "private"] = Field(
        description=(
            "Classification: 'public' (no private content), "
            "'mixed' (both public and private), 'private' (entirely private)"
        )
    )
    reasoning: str = Field(description="Brief explanation of the privacy classification decision")


def detect_privacy(chunk: Chunk, keywords: list[str], llm_client: ChatOpenAI) -> PrivacyResult:
    """Detect privacy level of a chunk using LLM with structured output.

    Args:
        chunk: The chunk to analyze.
        keywords: List of private keywords for context.
        llm_client: OpenRouter LLM client.

    Returns:
        Privacy detection result.
    """
    keywords_str = "\n".join(f"- {kw}" for kw in keywords)

    prompt = f"""You are analyzing a chunk from a document (cookbook or research paper).

Determine if this chunk contains private/restricted information.

**Private indicators include**:
{keywords_str}

Also look for:
- Explicit markers like "Restricted Section →", "Internal", "Secret"
- Methodology sections in research papers
- Internal financial data (salaries, costs)
- Staff information or conflicts

**Chunk text**:
{chunk.text}

Classify the privacy level:
- "public": No private content
- "mixed": Contains both public and private content
- "private": Entirely private content"""

    try:
        structured_llm = llm_client.with_structured_output(PrivacyResult)
        result: PrivacyResult = structured_llm.invoke(prompt)  # type: ignore[assignment]
        return result
    except Exception as e:
        logger.warning(f"Failed to detect privacy, defaulting to public: {e}")
        return PrivacyResult(
            has_private_info=False,
            privacy_level="public",
            reasoning="Error in detection, defaulted to public",
        )
