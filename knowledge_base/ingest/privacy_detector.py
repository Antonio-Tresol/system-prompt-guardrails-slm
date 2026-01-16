"""Privacy detection for document chunks.

This module provides privacy classification using direct LLM calls with JSON output.
Simple, fast, and robust - no complex agent/tool calling overhead.
"""

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from knowledge_base.ingest.chunkers import Chunk
from utils.logging import logger


class PrivacyResult(BaseModel):
    """Result of privacy detection."""

    has_private_info: bool = Field(
        description="True if the chunk contains any private/restricted information",
    )
    privacy_level: Literal["public", "mixed", "private"] = Field(
        description=(
            "Classification: 'public' (no private content), "
            "'mixed' (both public and private), 'private' (entirely private)"
        ),
    )
    reasoning: str = Field(description="Brief explanation of the privacy classification decision")


def _build_system_prompt() -> str:
    """Build system prompt with schema from PrivacyResult model."""
    schema = PrivacyResult.model_json_schema()
    return f"""You are a privacy classifier. Analyze text and classify its privacy level.
Respond with ONLY a JSON object (no markdown, no explanation before or after).

Required JSON schema:
{json.dumps(schema, indent=2)}

Rules:
- "public": No private/restricted content
- "mixed": Contains both public and private content
- "private": Entirely private/restricted content"""


def _build_user_message(*, chunk: Chunk, keywords: list[str]) -> str:
    """Build user message for privacy detection."""
    keywords_str = ", ".join(keywords) if keywords else "restricted, internal, secret, salary"
    return f"""Private keywords: {keywords_str}

Text to classify:
{chunk.text[:3000]}

Respond with ONLY the JSON object."""


def _get_default_result() -> PrivacyResult:
    """Return default public result for error cases."""
    return PrivacyResult(
        has_private_info=False,
        privacy_level="public",
        reasoning="Error in detection, defaulted to public",
    )


def _parse_json_from_text(text: str) -> dict | None:
    """Extract and parse JSON from model response text."""
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    return None


def _parse_response_to_result(response_text: str) -> PrivacyResult:
    """Parse model response text into PrivacyResult."""
    parsed = _parse_json_from_text(response_text)
    if parsed:
        privacy_level = parsed.get("privacy_level", "public")
        if privacy_level not in ["public", "mixed", "private"]:
            privacy_level = "public"

        return PrivacyResult(
            has_private_info=parsed.get("has_private_info", False),
            privacy_level=privacy_level,  # type: ignore[arg-type]
            reasoning=parsed.get("reasoning", "Classified successfully"),
        )
    return _get_default_result()


def detect_privacy(*, chunk: Chunk, keywords: list[str], model: ChatOpenAI) -> PrivacyResult:
    """Detect privacy level of a chunk using direct LLM call.

    Args:
        chunk: The chunk to analyze.
        keywords: List of private keywords for context.
        model: OpenRouter model client.

    Returns:
        Privacy detection result (never fails, returns default on error).
    """
    try:
        messages = [
            SystemMessage(content=_build_system_prompt()),
            HumanMessage(content=_build_user_message(chunk=chunk, keywords=keywords)),
        ]

        response = model.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        return _parse_response_to_result(str(response_text))
    except Exception as e:
        logger.warning(f"Privacy detection failed: {e}")
        return _get_default_result()


async def detect_privacy_async(
    *,
    chunk: Chunk,
    keywords: list[str],
    model: ChatOpenAI,
) -> PrivacyResult:
    """Async version of detect_privacy for parallel processing.

    Args:
        chunk: The chunk to analyze.
        keywords: List of private keywords for context.
        model: OpenRouter model client.

    Returns:
        Privacy detection result (never fails, returns default on error).
    """
    try:
        messages = [
            SystemMessage(content=_build_system_prompt()),
            HumanMessage(content=_build_user_message(chunk=chunk, keywords=keywords)),
        ]

        response = await model.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        return _parse_response_to_result(str(response_text))
    except Exception as e:
        logger.warning(f"Privacy detection failed: {e}")
        return _get_default_result()


async def detect_privacy_batch(
    *,
    chunks: list[Chunk],
    keywords: list[str],
    model: ChatOpenAI,
    max_concurrency: int = 10,
) -> list[PrivacyResult]:
    """Detect privacy for multiple chunks in parallel.

    Args:
        chunks: List of chunks to analyze.
        keywords: List of private keywords for context.
        model: OpenRouter model client.
        max_concurrency: Maximum number of concurrent API calls.

    Returns:
        List of privacy detection results in same order as input chunks.
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_detect(chunk: Chunk) -> PrivacyResult:
        async with semaphore:
            return await detect_privacy_async(chunk=chunk, keywords=keywords, model=model)

    tasks = [bounded_detect(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_results: list[PrivacyResult] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Chunk {i} privacy detection failed: {result}")
            final_results.append(_get_default_result())
        else:
            final_results.append(result)

    return final_results
