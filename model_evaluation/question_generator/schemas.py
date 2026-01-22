"""Schemas for question generation.

This module defines the MetaQuestion dataclass used for generated questions.
"""

from dataclasses import dataclass


@dataclass
class MetaQuestion:
    """A generated test question with metadata.

    Attributes:
        question: The actual question text to ask the model.
        universe_context: Name of the universe context (e.g., 'moonlit_granary').
        is_refusal: True if the question should be refused (requests private info).
    """

    question: str
    universe_context: str
    is_refusal: bool
