"""Knowledge Base Generator Agent package.

This package provides an LLM-based agent that generates synthetic knowledge base
documents at query time for testing AI safety evaluation. It replaces the static
vector store approach with dynamic generation to ensure research integrity.

Key components:
- GeneratorContext: Dataclass for runtime context (include_private_info flag)
- DocumentChunk: Pydantic model for individual document chunks
- GeneratorOutput: Pydantic model for structured output (1-3 chunks)
- GeneratorSession: Session manager maintaining history across queries
- create_kb_generator_agent: Factory function to create the generator agent
"""

from model_evaluation.main_agent.kb_generator.agent import create_kb_generator_agent
from model_evaluation.main_agent.kb_generator.schemas import (
    DocumentChunk,
    GeneratorContext,
    GeneratorOutput,
)
from model_evaluation.main_agent.kb_generator.session import GeneratorSession

__all__ = [
    "create_kb_generator_agent",
    "DocumentChunk",
    "GeneratorContext",
    "GeneratorOutput",
    "GeneratorSession",
]
