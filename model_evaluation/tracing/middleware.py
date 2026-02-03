"""Trajectory capture middleware for the safety evaluation agent.

Uses LangChain's AgentMiddleware hooks to record the full agent
trajectory (model calls, tool executions, token counts) as structured
Python objects without any external service dependency.
"""

import time
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from model_evaluation.main_agent.gemma_wrapper import GemmaWithSAE
from model_evaluation.main_agent.tools import EvaluationContext
from model_evaluation.tracing.schemas import AgentStep, AgentTrace, ToolCallTrace


def _serialize_message(msg: BaseMessage | object) -> dict[str, Any]:
    """Serialize a LangChain message to a JSON-compatible dict.

    Args:
        msg: A LangChain message object.

    Returns:
        Dictionary representation of the message.
    """
    if hasattr(msg, "model_dump"):
        return msg.model_dump(mode="json")
    if hasattr(msg, "dict"):
        return msg.dict()
    return {"content": str(msg), "type": type(msg).__name__}


class TrajectoryCapture(AgentMiddleware[AgentState, EvaluationContext]):
    """Middleware that captures the full agent trajectory during execution.

    Records every model invocation, tool call, and token count into an
    AgentTrace object. Traces accumulate across multiple agent runs and
    can be retrieved via `last_trace` or `all_traces`.

    Attributes:
        state_schema: The agent state schema type.
        tools: Additional tools registered by this middleware (none).
    """

    state_schema = AgentState
    tools: list = []  # type: ignore[assignment]

    def __init__(self) -> None:
        """Initialize the trajectory capture middleware."""
        self._current_trace: AgentTrace | None = None
        self._current_step: AgentStep | None = None
        self._step_counter: int = 0
        self._start_time: float = 0.0
        self._tokens_before_input: int = 0
        self._tokens_before_output: int = 0
        self._completed_traces: list[AgentTrace] = []

    @property
    def last_trace(self) -> AgentTrace | None:
        """Return the most recently completed trace.

        Returns:
            The last completed AgentTrace, or None if no traces have been recorded.
        """
        if self._completed_traces:
            return self._completed_traces[-1]
        return None

    @property
    def all_traces(self) -> list[AgentTrace]:
        """Return all completed traces.

        Returns:
            List of all completed AgentTrace objects.
        """
        return list(self._completed_traces)

    def clear(self) -> None:
        """Clear all completed traces."""
        self._completed_traces.clear()

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime[EvaluationContext],
    ) -> dict[str, Any] | None:
        """Initialize a new trace at the start of an agent run.

        Args:
            state: Current agent state with messages.
            runtime: Runtime context with EvaluationContext.

        Returns:
            None (no state modifications).
        """
        self._current_trace = AgentTrace()
        self._current_step = None
        self._step_counter = 0
        self._start_time = time.perf_counter()

        messages = state.get("messages")
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                self._current_trace.question_text = str(last_msg.content)

        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Capture model invocation details and response.

        Records the messages snapshot, calls the model, then records
        the response content, tool calls, and token counts.

        Args:
            request: The model request with messages, system prompt, and tools.
            handler: Callback to execute the actual model call.

        Returns:
            The unmodified ModelResponse from the handler.
        """
        if self._current_trace is None:
            return handler(request)

        self._step_counter += 1

        # Capture system prompt on first step
        if self._step_counter == 1:
            if request.system_message is not None:
                self._current_trace.system_prompt = str(request.system_message.content)
            self._current_trace.available_tools = [
                t.name if hasattr(t, "name") else str(t) for t in request.tools
            ]

        # Build messages snapshot
        messages_snapshot: list[dict[str, Any]] = []
        if request.system_message is not None:
            messages_snapshot.append(_serialize_message(request.system_message))
        for msg in request.messages:
            messages_snapshot.append(_serialize_message(msg))

        # Record token counts before call
        model = request.model
        if isinstance(model, GemmaWithSAE):
            self._tokens_before_input = model.total_input_tokens
            self._tokens_before_output = model.total_output_tokens

        # Execute the model call
        response = handler(request)

        # Calculate per-step token deltas
        input_tokens = 0
        output_tokens = 0
        if isinstance(model, GemmaWithSAE):
            input_tokens = model.total_input_tokens - self._tokens_before_input
            output_tokens = model.total_output_tokens - self._tokens_before_output

        # Extract response content and tool calls
        response_content: str | None = None
        response_tool_calls: list[dict[str, Any]] = []

        for msg in response.result:
            if isinstance(msg, AIMessage):
                if msg.content:
                    response_content = str(msg.content)
                if msg.tool_calls:
                    response_tool_calls = [
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", ""),
                        }
                        for tc in msg.tool_calls
                    ]

        # Create the step record
        self._current_step = AgentStep(
            step_number=self._step_counter,
            messages_snapshot=messages_snapshot,
            model_response_content=response_content,
            model_response_tool_calls=response_tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._current_trace.steps.append(self._current_step)

        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Capture tool execution details and result.

        Args:
            request: The tool call request with tool name, args, and ID.
            handler: Callback to execute the actual tool call.

        Returns:
            The unmodified ToolMessage or Command from the handler.
        """
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown")
        tool_call_id = tool_call.get("id", "")
        arguments = tool_call.get("args", {})

        start = time.perf_counter()
        result = handler(request)
        duration_ms = (time.perf_counter() - start) * 1000

        result_content = ""
        if isinstance(result, ToolMessage) and result.content:
            result_content = str(result.content)

        tool_trace = ToolCallTrace(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            result=result_content,
            duration_ms=duration_ms,
        )

        if self._current_step is not None:
            self._current_step.tool_executions.append(tool_trace)

        return result

    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime[EvaluationContext],
    ) -> dict[str, Any] | None:
        """Finalize the trace after agent execution completes.

        Computes aggregate metrics and stores the completed trace.

        Args:
            state: Final agent state with all messages.
            runtime: Runtime context.

        Returns:
            None (no state modifications).
        """
        if self._current_trace is None:
            return None

        self._current_trace.total_duration_ms = (time.perf_counter() - self._start_time) * 1000

        self._current_trace.finalize()
        self._completed_traces.append(self._current_trace)
        self._current_trace = None
        self._current_step = None

        return None
