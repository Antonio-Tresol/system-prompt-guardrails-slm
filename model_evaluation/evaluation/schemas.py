"""Schemas for the evaluation pipeline."""

from typing import Literal

from pydantic import BaseModel, Field

DISPLAY_NAME_TO_YAML_KEY: dict[str, str | None] = {
    "The Carnelian Table": "carnelian_table",
    "Brine & Riddle": "brine_and_riddle",
    "The Moonlit Granary": "moonlit_granary",
    "Velvet Hourglass": "velvet_hourglass",
    "All": None,
}


class QuestionRow(BaseModel):
    """A single question parsed from the evaluation CSV."""

    number: int
    question: str
    document_of_origin: str
    is_malicious: bool
    universe_context_key: str | None = Field(
        description="YAML key for the universe context, or None for 'All' questions.",
    )


class RunResult(BaseModel):
    """Result of a single (question, prompt_format) evaluation run."""

    question_id: int
    question_text: str
    expects_refusal: bool
    universe_context: str | None
    prompt_format: Literal["markdown", "plain"]
    model_size: str
    final_answer: str
    num_steps: int
    num_tool_calls: int
    tool_names: list[str]
    total_input_tokens: int
    total_output_tokens: int
    duration_ms: float
    trace_id: str
    sae_l0_by_layer: dict[int, float] = Field(default_factory=dict)
    sae_fvu_by_layer: dict[int, float] = Field(default_factory=dict)
