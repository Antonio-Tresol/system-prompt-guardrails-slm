import json
import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from knowledge_base.ingest.chunkers import Chunk

logger = logging.getLogger(__name__)


class PrivacyResult(BaseModel):
    """Result of privacy detection."""

    has_private_info: bool
    privacy_level: Literal["public", "mixed", "private"]
    reasoning: str


def detect_privacy(chunk: Chunk, keywords: list[str], llm_client: ChatOpenAI) -> PrivacyResult:
    """Detect privacy level of a chunk using LLM.

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
- "private": Entirely private content

Respond ONLY with JSON:
{{"has_private_info": bool, "privacy_level": "public|mixed|private", "reasoning": str}}"""

    try:
        response = llm_client.invoke(prompt)
        content = response.content

        if isinstance(content, list) and content and isinstance(content[0], dict):
            content = content[0].get("text", "{}")
        elif not isinstance(content, str):
            content = str(content)

        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        result_dict = json.loads(content)
        return PrivacyResult(**result_dict)
    except Exception as e:
        logger.warning(f"Failed to detect privacy, defaulting to public: {e}")
        return PrivacyResult(
            has_private_info=False, privacy_level="public", reasoning="Error in detection"
        )
