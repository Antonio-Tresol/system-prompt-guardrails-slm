"""Question Generator package for safety evaluation.

This package generates evaluation questions grounded in synthetic cookbook documents.
Questions are categorized as either expecting a refusal (private info request) or
expecting an answer (public info request).
"""

from data_generation.question_generator.agent import (
    create_question_generator_agent,
    generate_questions,
)
from data_generation.question_generator.schemas import (
    GeneratedQuestion,
    QuestionBatch,
    QuestionGeneratorContext,
)

__all__ = [
    "GeneratedQuestion",
    "QuestionBatch",
    "QuestionGeneratorContext",
    "create_question_generator_agent",
    "generate_questions",
]
