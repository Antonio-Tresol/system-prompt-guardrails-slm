"""Schemas for the Question Generator Agent."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class QuestionGeneratorContext:
    """Context passed to the question generator agent.

    Contains the universe context (YAML content) and configuration for
    what types of questions to generate.
    """

    universe_context: str
    document_name: str
    question_type: Literal["private", "public", "both"]


class GeneratedQuestion(BaseModel):
    """A single generated evaluation question.

    Represents a question that can be used to evaluate whether an agent
    properly handles public vs private information.
    """

    question: str = Field(
        description="The natural language question a user might ask",
    )
    expects_refusal: bool = Field(
        description="True if the question asks for private info (agent should refuse)",
    )
    target_info_type: str = Field(
        description="What type of info the question targets (e.g., 'salary', 'secret_ingredient')",
    )
    grounding_elements: list[str] = Field(
        description="Specific names/items from the universe used in this question",
    )


class QuestionBatch(BaseModel):
    """Batch of generated questions from a single generation call.

    Contains multiple questions ensuring variety in question types and targets.
    """

    questions: list[GeneratedQuestion] = Field(
        description="5-10 generated questions per batch",
        min_length=1,
        max_length=15,
    )


# Mapping of document identifiers to their YAML file paths
DOCUMENT_YAML_MAP: dict[str, Path] = {
    "carnelian_table": Path("data_generation/universe_contexts/carnelian_table.yaml"),
    "brine_and_riddle": Path("data_generation/universe_contexts/brine_and_riddle.yaml"),
    "moonlit_granary": Path("data_generation/universe_contexts/moonlit_granary.yaml"),
    "velvet_hourglass": Path("data_generation/universe_contexts/velvet_hourglass.yaml"),
}

# Mapping of document identifiers to their display names (for CSV)
DOCUMENT_DISPLAY_NAMES: dict[str, str] = {
    "carnelian_table": "The Carnelian Table",
    "brine_and_riddle": "Brine & Riddle",
    "moonlit_granary": "The Moonlit Granary",
    "velvet_hourglass": "Velvet Hourglass",
}
