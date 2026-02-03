"""Pydantic schemas for agent trajectory tracing.

Defines the data model for capturing full agent execution traces
as structured Python objects that serialize to JSON.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallTrace(BaseModel):
    """A single tool call within an agent step.

    Attributes:
        tool_name: Name of the tool that was called.
        tool_call_id: Unique ID linking this call to the model's tool_call request.
        arguments: Arguments passed to the tool.
        result: String result returned by the tool.
        duration_ms: Time spent executing the tool in milliseconds.
    """

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    result: str
    duration_ms: float


class AgentStep(BaseModel):
    """A single model invocation step within the agent loop.

    Each step represents one call to the model, including the full
    message context sent, the model's response, and any tool calls
    that were executed as a result.

    Attributes:
        step_number: Sequential step index (1-based).
        messages_snapshot: All messages sent to the model at this step.
        model_response_content: Text content of the model's response.
        model_response_tool_calls: Tool calls requested by the model.
        tool_executions: Results of tool calls executed after this model response.
        input_tokens: Number of input tokens for this step.
        output_tokens: Number of output tokens for this step.
    """

    step_number: int
    messages_snapshot: list[dict[str, Any]]
    model_response_content: str | None = None
    model_response_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_executions: list[ToolCallTrace] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class AgentTrace(BaseModel):
    """Complete trace of a single agent run.

    Captures the full trajectory from user question to final answer,
    including every model call, tool execution, and token count.

    Attributes:
        trace_id: Unique identifier for this trace.
        timestamp: ISO 8601 timestamp of when the trace started.
        question_text: The user's input question.
        question_id: Optional question ID from the question bank.
        is_refusal: Ground truth label (True=should refuse, False=should answer).
        universe_context: Name of the universe context used.
        system_prompt_format: Whether Markdown or Plain Text prompt was used.
        turn_type: Whether this is a single-turn or multi-turn interaction.
        system_prompt: The full system prompt text.
        available_tools: Names of tools available to the agent.
        steps: Ordered list of agent steps in the trajectory.
        final_answer: The agent's final text response.
        total_steps: Number of model invocations.
        total_tool_calls: Total number of tool executions across all steps.
        total_input_tokens: Sum of input tokens across all steps.
        total_output_tokens: Sum of output tokens across all steps.
        total_duration_ms: Wall-clock time for the entire agent run.
    """

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(),
    )

    question_text: str = ""
    question_id: str | None = None
    is_refusal: bool | None = None
    universe_context: str | None = None
    system_prompt_format: Literal["markdown", "plain"] = "markdown"
    turn_type: Literal["single", "multi"] = "single"

    system_prompt: str = ""
    available_tools: list[str] = Field(default_factory=list)

    steps: list[AgentStep] = Field(default_factory=list)

    final_answer: str | None = None

    total_steps: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_ms: float = 0.0

    def finalize(self) -> None:
        """Compute aggregate metrics from the recorded steps."""
        self.total_steps = len(self.steps)
        self.total_tool_calls = sum(len(step.tool_executions) for step in self.steps)
        self.total_input_tokens = sum(step.input_tokens for step in self.steps)
        self.total_output_tokens = sum(step.output_tokens for step in self.steps)

        if self.steps:
            last_step = self.steps[-1]
            if last_step.model_response_content and not last_step.model_response_tool_calls:
                self.final_answer = last_step.model_response_content
