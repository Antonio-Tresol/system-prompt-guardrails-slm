"""Schemas for the evaluation pipeline."""

from typing import Literal

from pydantic import BaseModel, Field

DISPLAY_NAME_TO_YAML_KEY: dict[str, str] = {
    "The Carnelian Table": "the_carnelian_table",
    "Hartwell & Grey": "hartwell_and_grey",
    "Linden Grove Clinic": "linden_grove_clinic",
    "Nova Circuit Labs": "nova_circuit_labs",
}


class QuestionRow(BaseModel):
    """A single question parsed from the evaluation CSV."""

    number: int
    question: str
    universe: str
    is_malicious: bool
    universe_context_key: str = Field(
        description="YAML key for the universe context.",
    )


class RunResult(BaseModel):
    """Result of a single (question, prompt_format) evaluation run."""

    question_id: int
    question_text: str
    expects_refusal: bool
    universe_context: str
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

    # Judge classification (refusal)
    model_refused: bool | None = Field(
        default=None,
        description="Whether the LLM judge classified the response as a refusal.",
    )
    judge_reasoning: str | None = Field(
        default=None,
        description="Judge's explanation for the classification.",
    )
    judge_error: str | None = Field(
        default=None,
        description="Error message if judge classification failed.",
    )

    # Groundedness classification
    kb_sources: str | None = Field(
        default=None,
        description="Formatted KB sources the agent received from search_knowledge_base.",
    )
    is_grounded: bool | None = Field(
        default=None,
        description="Whether the response is grounded in KB sources (LLM judge).",
    )
    groundedness_reasoning: str | None = Field(
        default=None,
        description="Judge's reasoning for groundedness classification.",
    )
    groundedness_error: str | None = Field(
        default=None,
        description="Error message if groundedness judge failed.",
    )
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts needed due to hallucination (0 = first try).",
    )

    sae_l0_by_layer: dict[int, float] = Field(default_factory=dict)
    sae_fvu_by_layer: dict[int, float] = Field(default_factory=dict)
