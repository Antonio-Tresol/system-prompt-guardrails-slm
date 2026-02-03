"""Tests for trace schema serialization and validation."""

import json

from model_evaluation.tracing.schemas import AgentStep, AgentTrace, ToolCallTrace


class TestToolCallTrace:
    """Tests for ToolCallTrace schema."""

    def test_serialization_round_trip(self) -> None:
        """ToolCallTrace serializes to JSON and deserializes back identically."""
        trace = ToolCallTrace(
            tool_name="search_knowledge_base",
            tool_call_id="call_123",
            arguments={"query": "menu items", "num_results": 5},
            result="--- Result 1 [Public] ---\nContent: ...",
            duration_ms=150.5,
        )

        json_str = trace.model_dump_json()
        restored = ToolCallTrace.model_validate_json(json_str)

        assert restored == trace

    def test_serialization_with_nested_arguments(self) -> None:
        """ToolCallTrace handles complex nested argument structures."""
        trace = ToolCallTrace(
            tool_name="think",
            tool_call_id="call_456",
            arguments={"thought": "Let me analyze the query"},
            result="Thought recorded: Let me analyze the query",
            duration_ms=0.1,
        )

        data = trace.model_dump(mode="json")
        assert data["arguments"]["thought"] == "Let me analyze the query"


class TestAgentStep:
    """Tests for AgentStep schema."""

    def test_default_values(self) -> None:
        """AgentStep has sensible defaults for optional fields."""
        step = AgentStep(
            step_number=1,
            messages_snapshot=[{"role": "user", "content": "hello"}],
        )

        assert step.model_response_content is None
        assert step.model_response_tool_calls == []
        assert step.tool_executions == []
        assert step.input_tokens == 0
        assert step.output_tokens == 0

    def test_full_step_serialization(self) -> None:
        """AgentStep with all fields populated serializes correctly."""
        tool_trace = ToolCallTrace(
            tool_name="think",
            tool_call_id="call_1",
            arguments={"thought": "planning"},
            result="Thought recorded: planning",
            duration_ms=0.5,
        )
        step = AgentStep(
            step_number=2,
            messages_snapshot=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is the menu?"},
                {"role": "assistant", "content": "", "tool_calls": [{"name": "think"}]},
            ],
            model_response_content=None,
            model_response_tool_calls=[{"name": "think", "args": {"thought": "planning"}}],
            tool_executions=[tool_trace],
            input_tokens=100,
            output_tokens=25,
        )

        json_str = step.model_dump_json()
        restored = AgentStep.model_validate_json(json_str)

        assert restored.step_number == 2
        assert len(restored.messages_snapshot) == 3
        assert len(restored.tool_executions) == 1
        assert restored.tool_executions[0].tool_name == "think"


class TestAgentTrace:
    """Tests for AgentTrace schema."""

    def test_default_initialization(self) -> None:
        """AgentTrace initializes with auto-generated trace_id and timestamp."""
        trace = AgentTrace()

        assert trace.trace_id  # non-empty UUID string
        assert trace.timestamp  # non-empty ISO timestamp
        assert trace.steps == []
        assert trace.final_answer is None
        assert trace.total_steps == 0

    def test_finalize_computes_aggregates(self) -> None:
        """finalize() correctly computes aggregate metrics from steps."""
        tool1 = ToolCallTrace(
            tool_name="think",
            tool_call_id="c1",
            arguments={"thought": "a"},
            result="ok",
            duration_ms=1.0,
        )
        tool2 = ToolCallTrace(
            tool_name="search_knowledge_base",
            tool_call_id="c2",
            arguments={"query": "menu"},
            result="results",
            duration_ms=200.0,
        )
        step1 = AgentStep(
            step_number=1,
            messages_snapshot=[],
            model_response_content=None,
            model_response_tool_calls=[{"name": "think"}],
            tool_executions=[tool1],
            input_tokens=50,
            output_tokens=10,
        )
        step2 = AgentStep(
            step_number=2,
            messages_snapshot=[],
            model_response_content=None,
            model_response_tool_calls=[{"name": "search_knowledge_base"}],
            tool_executions=[tool2],
            input_tokens=80,
            output_tokens=15,
        )
        step3 = AgentStep(
            step_number=3,
            messages_snapshot=[],
            model_response_content="Here is the menu information.",
            model_response_tool_calls=[],
            input_tokens=120,
            output_tokens=30,
        )

        trace = AgentTrace(steps=[step1, step2, step3])
        trace.finalize()

        assert trace.total_steps == 3
        assert trace.total_tool_calls == 2
        assert trace.total_input_tokens == 250
        assert trace.total_output_tokens == 55
        assert trace.final_answer == "Here is the menu information."

    def test_finalize_no_final_answer_when_last_step_has_tool_calls(self) -> None:
        """finalize() does not set final_answer if the last step made tool calls."""
        step = AgentStep(
            step_number=1,
            messages_snapshot=[],
            model_response_content="I'll search for that.",
            model_response_tool_calls=[{"name": "search_knowledge_base"}],
            input_tokens=50,
            output_tokens=10,
        )

        trace = AgentTrace(steps=[step])
        trace.finalize()

        assert trace.final_answer is None

    def test_full_trace_json_round_trip(self) -> None:
        """A complete AgentTrace survives JSON serialization and deserialization."""
        trace = AgentTrace(
            question_text="What are the prices?",
            question_id="s3",
            is_refusal=False,
            universe_context="carnelian_table",
            system_prompt_format="markdown",
            turn_type="single",
            system_prompt="# Safety Assistant\nYou are helpful.",
            available_tools=["think", "search_knowledge_base"],
            steps=[
                AgentStep(
                    step_number=1,
                    messages_snapshot=[{"role": "user", "content": "What are the prices?"}],
                    model_response_content="The appetizers start at $12.",
                    input_tokens=100,
                    output_tokens=20,
                ),
            ],
            total_duration_ms=1500.0,
        )
        trace.finalize()

        json_str = trace.model_dump_json(indent=2)
        data = json.loads(json_str)

        assert data["question_text"] == "What are the prices?"
        assert data["system_prompt_format"] == "markdown"
        assert len(data["steps"]) == 1
        assert data["total_steps"] == 1

        restored = AgentTrace.model_validate_json(json_str)
        assert restored.question_text == trace.question_text
        assert restored.final_answer == trace.final_answer

    def test_finalize_empty_trace(self) -> None:
        """finalize() handles an empty trace with no steps."""
        trace = AgentTrace()
        trace.finalize()

        assert trace.total_steps == 0
        assert trace.total_tool_calls == 0
        assert trace.total_input_tokens == 0
        assert trace.total_output_tokens == 0
        assert trace.final_answer is None
