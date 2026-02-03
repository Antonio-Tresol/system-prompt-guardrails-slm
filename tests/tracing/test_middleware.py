"""Tests for TrajectoryCapture middleware hook behavior."""

from unittest.mock import MagicMock, patch

from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from model_evaluation.tracing.middleware import TrajectoryCapture


def _make_agent_state(*, messages: list | None = None) -> AgentState:
    """Create a minimal AgentState for testing.

    Args:
        messages: Optional list of messages.

    Returns:
        An AgentState dict.
    """
    return {"messages": messages or []}


def _make_runtime() -> MagicMock:
    """Create a mock Runtime object.

    Returns:
        A MagicMock mimicking Runtime[EvaluationContext].
    """
    runtime = MagicMock()
    runtime.context = MagicMock()
    return runtime


class TestTrajectoryCapture:
    """Tests for the TrajectoryCapture middleware class."""

    def test_initial_state(self) -> None:
        """Middleware starts with no traces."""
        middleware = TrajectoryCapture()

        assert middleware.last_trace is None
        assert middleware.all_traces == []

    def test_before_agent_initializes_trace(self) -> None:
        """before_agent creates a new trace with the question text."""
        middleware = TrajectoryCapture()
        user_msg = MagicMock()
        user_msg.content = "What is on the menu?"
        state = _make_agent_state(messages=[user_msg])

        result = middleware.before_agent(state, _make_runtime())

        assert result is None
        assert middleware._current_trace is not None
        assert middleware._current_trace.question_text == "What is on the menu?"

    def test_wrap_model_call_captures_step(self) -> None:
        """wrap_model_call records a step with messages, response, and tokens."""
        middleware = TrajectoryCapture()

        # Initialize trace
        user_msg = MagicMock()
        user_msg.content = "Hello"
        middleware.before_agent(_make_agent_state(messages=[user_msg]), _make_runtime())

        # Build a model request
        model_mock = MagicMock()
        model_mock.total_input_tokens = 0
        model_mock.total_output_tokens = 0

        # Patch isinstance check for GemmaWithSAE
        sys_msg = SystemMessage(content="You are helpful.")
        human_msg = MagicMock()
        human_msg.model_dump.return_value = {"role": "user", "content": "Hello"}

        request = MagicMock(spec=ModelRequest)
        request.model = model_mock
        request.system_message = sys_msg
        request.messages = [human_msg]
        request.tools = []

        # Handler returns an AIMessage
        ai_msg = AIMessage(content="Hi there!", tool_calls=[])
        response = ModelResponse(result=[ai_msg])

        def handler(req: ModelRequest) -> ModelResponse:
            model_mock.total_input_tokens = 50
            model_mock.total_output_tokens = 10
            return response

        with patch(
            "model_evaluation.tracing.middleware.isinstance",
            side_effect=lambda obj, cls: True if obj is model_mock else isinstance(obj, cls),
        ):
            result = middleware.wrap_model_call(request, handler)

        assert result is response
        assert len(middleware._current_trace.steps) == 1

        step = middleware._current_trace.steps[0]
        assert step.step_number == 1
        assert step.model_response_content == "Hi there!"
        assert step.model_response_tool_calls == []
        assert len(step.messages_snapshot) == 2  # system + user

    def test_wrap_model_call_captures_tool_calls(self) -> None:
        """wrap_model_call records tool calls from the model response."""
        middleware = TrajectoryCapture()
        user_msg = MagicMock()
        user_msg.content = "Search something"
        middleware.before_agent(_make_agent_state(messages=[user_msg]), _make_runtime())

        model_mock = MagicMock()
        model_mock.total_input_tokens = 0
        model_mock.total_output_tokens = 0

        request = MagicMock(spec=ModelRequest)
        request.model = model_mock
        request.system_message = None
        request.messages = []
        request.tools = [MagicMock(name="think"), MagicMock(name="search_knowledge_base")]

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "think", "args": {"thought": "planning"}, "id": "call_1"},
            ],
        )
        response = ModelResponse(result=[ai_msg])

        middleware.wrap_model_call(request, lambda r: response)

        step = middleware._current_trace.steps[0]
        assert len(step.model_response_tool_calls) == 1
        assert step.model_response_tool_calls[0]["name"] == "think"

    def test_wrap_tool_call_captures_execution(self) -> None:
        """wrap_tool_call records tool name, args, result, and duration."""
        middleware = TrajectoryCapture()
        user_msg = MagicMock()
        user_msg.content = "Q"
        middleware.before_agent(_make_agent_state(messages=[user_msg]), _make_runtime())

        # Create a model step first (tool executions attach to current step)
        request = MagicMock(spec=ModelRequest)
        request.model = MagicMock()
        request.model.total_input_tokens = 0
        request.model.total_output_tokens = 0
        request.system_message = None
        request.messages = []
        request.tools = []

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "think", "args": {"thought": "x"}, "id": "c1"}],
        )
        middleware.wrap_model_call(request, lambda r: ModelResponse(result=[ai_msg]))

        # Now wrap the tool call
        tool_request = MagicMock(spec=ToolCallRequest)
        tool_request.tool_call = {
            "name": "think",
            "args": {"thought": "analyzing"},
            "id": "call_99",
        }

        tool_result = ToolMessage(
            content="Thought recorded: analyzing",
            tool_call_id="call_99",
            name="think",
        )

        result = middleware.wrap_tool_call(tool_request, lambda r: tool_result)

        assert result is tool_result
        assert len(middleware._current_trace.steps[0].tool_executions) == 1

        tool_trace = middleware._current_trace.steps[0].tool_executions[0]
        assert tool_trace.tool_name == "think"
        assert tool_trace.tool_call_id == "call_99"
        assert tool_trace.arguments == {"thought": "analyzing"}
        assert "Thought recorded" in tool_trace.result
        assert tool_trace.duration_ms >= 0

    def test_after_agent_finalizes_trace(self) -> None:
        """after_agent computes aggregates and stores the completed trace."""
        middleware = TrajectoryCapture()
        user_msg = MagicMock()
        user_msg.content = "Q"
        middleware.before_agent(_make_agent_state(messages=[user_msg]), _make_runtime())

        # Simulate a model call with a final answer
        request = MagicMock(spec=ModelRequest)
        request.model = MagicMock()
        request.model.total_input_tokens = 0
        request.model.total_output_tokens = 0
        request.system_message = None
        request.messages = []
        request.tools = []

        ai_msg = AIMessage(content="The answer is 42.", tool_calls=[])
        middleware.wrap_model_call(request, lambda r: ModelResponse(result=[ai_msg]))

        # Finalize
        result = middleware.after_agent(_make_agent_state(), _make_runtime())

        assert result is None
        assert middleware._current_trace is None
        assert middleware.last_trace is not None
        assert middleware.last_trace.total_steps == 1
        assert middleware.last_trace.final_answer == "The answer is 42."

    def test_multiple_runs_accumulate_traces(self) -> None:
        """Running the agent multiple times accumulates traces."""
        middleware = TrajectoryCapture()
        runtime = _make_runtime()

        for i in range(3):
            user_msg = MagicMock()
            user_msg.content = f"Question {i}"
            middleware.before_agent(_make_agent_state(messages=[user_msg]), runtime)

            request = MagicMock(spec=ModelRequest)
            request.model = MagicMock()
            request.model.total_input_tokens = 0
            request.model.total_output_tokens = 0
            request.system_message = None
            request.messages = []
            request.tools = []

            ai_msg = AIMessage(content=f"Answer {i}", tool_calls=[])
            response = ModelResponse(result=[ai_msg])
            middleware.wrap_model_call(request, lambda r, resp=response: resp)
            middleware.after_agent(_make_agent_state(), runtime)

        assert len(middleware.all_traces) == 3
        assert middleware.last_trace.question_text == "Question 2"

    def test_clear_removes_all_traces(self) -> None:
        """clear() removes all completed traces."""
        middleware = TrajectoryCapture()
        runtime = _make_runtime()

        user_msg = MagicMock()
        user_msg.content = "Q"
        middleware.before_agent(_make_agent_state(messages=[user_msg]), runtime)
        middleware.after_agent(_make_agent_state(), runtime)

        assert len(middleware.all_traces) == 1
        middleware.clear()
        assert len(middleware.all_traces) == 0
        assert middleware.last_trace is None

    def test_system_prompt_captured_on_first_step(self) -> None:
        """The system prompt is captured from the first model request."""
        middleware = TrajectoryCapture()
        user_msg = MagicMock()
        user_msg.content = "Q"
        middleware.before_agent(_make_agent_state(messages=[user_msg]), _make_runtime())

        request = MagicMock(spec=ModelRequest)
        request.model = MagicMock()
        request.model.total_input_tokens = 0
        request.model.total_output_tokens = 0
        request.system_message = SystemMessage(content="# Safety Assistant")
        request.messages = []
        request.tools = [MagicMock(name="think")]

        ai_msg = AIMessage(content="Hello", tool_calls=[])
        middleware.wrap_model_call(request, lambda r: ModelResponse(result=[ai_msg]))

        assert middleware._current_trace.system_prompt == "# Safety Assistant"

    def test_no_trace_passthrough(self) -> None:
        """wrap_model_call passes through when no trace is active."""
        middleware = TrajectoryCapture()

        request = MagicMock(spec=ModelRequest)
        ai_msg = AIMessage(content="hi", tool_calls=[])
        response = ModelResponse(result=[ai_msg])

        result = middleware.wrap_model_call(request, lambda r: response)

        assert result is response
