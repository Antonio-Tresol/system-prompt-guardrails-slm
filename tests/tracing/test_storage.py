"""Tests for trace storage utilities."""

import json
from pathlib import Path

import pytest

from model_evaluation.tracing.schemas import AgentStep, AgentTrace, ToolCallTrace
from model_evaluation.tracing.storage import save_trace, save_traces


@pytest.fixture
def sample_trace() -> AgentTrace:
    """Create a sample trace for testing.

    Returns:
        A populated AgentTrace instance.
    """
    tool = ToolCallTrace(
        tool_name="think",
        tool_call_id="c1",
        arguments={"thought": "planning"},
        result="Thought recorded: planning",
        duration_ms=1.2,
    )
    step = AgentStep(
        step_number=1,
        messages_snapshot=[{"role": "user", "content": "What is the menu?"}],
        model_response_content="Here are the menu items.",
        model_response_tool_calls=[],
        tool_executions=[tool],
        input_tokens=100,
        output_tokens=25,
    )
    trace = AgentTrace(
        question_text="What is the menu?",
        question_id="s1",
        is_refusal=False,
        universe_context="carnelian_table",
        system_prompt_format="markdown",
        turn_type="single",
        system_prompt="# Safety Assistant",
        available_tools=["think", "search_knowledge_base"],
        steps=[step],
        total_duration_ms=500.0,
    )
    trace.finalize()
    return trace


class TestSaveTrace:
    """Tests for save_trace function."""

    def test_save_creates_file(self, *, sample_trace: AgentTrace, tmp_path: Path) -> None:
        """save_trace creates a JSON file at the expected path."""
        filepath = save_trace(trace=sample_trace, output_dir=tmp_path)

        assert filepath.exists()
        assert filepath.suffix == ".json"
        assert sample_trace.trace_id in filepath.name

    def test_save_produces_valid_json(self, *, sample_trace: AgentTrace, tmp_path: Path) -> None:
        """save_trace produces valid JSON that can be parsed."""
        filepath = save_trace(trace=sample_trace, output_dir=tmp_path)
        data = json.loads(filepath.read_text(encoding="utf-8"))

        assert data["question_text"] == "What is the menu?"
        assert data["system_prompt_format"] == "markdown"
        assert len(data["steps"]) == 1
        assert data["total_steps"] == 1

    def test_save_creates_output_directory(
        self,
        *,
        sample_trace: AgentTrace,
        tmp_path: Path,
    ) -> None:
        """save_trace creates the output directory if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "output"
        filepath = save_trace(trace=sample_trace, output_dir=nested_dir)

        assert filepath.exists()
        assert nested_dir.exists()

    def test_save_round_trip(self, *, sample_trace: AgentTrace, tmp_path: Path) -> None:
        """A saved trace can be deserialized back to an identical AgentTrace."""
        filepath = save_trace(trace=sample_trace, output_dir=tmp_path)
        json_str = filepath.read_text(encoding="utf-8")
        restored = AgentTrace.model_validate_json(json_str)

        assert restored.trace_id == sample_trace.trace_id
        assert restored.question_text == sample_trace.question_text
        assert restored.total_steps == sample_trace.total_steps
        assert restored.final_answer == sample_trace.final_answer
        assert len(restored.steps) == len(sample_trace.steps)
        assert restored.steps[0].tool_executions[0].tool_name == "think"


class TestSaveTraces:
    """Tests for save_traces function."""

    def test_save_multiple_traces(self, *, sample_trace: AgentTrace, tmp_path: Path) -> None:
        """save_traces saves a list of traces as a JSON array."""
        traces = [sample_trace, sample_trace]
        filepath = save_traces(traces=traces, output_dir=tmp_path)

        assert filepath.exists()
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 2

    def test_save_with_custom_filename(
        self,
        *,
        sample_trace: AgentTrace,
        tmp_path: Path,
    ) -> None:
        """save_traces uses the provided filename."""
        filepath = save_traces(
            traces=[sample_trace],
            output_dir=tmp_path,
            filename="custom_output.json",
        )

        assert filepath.name == "custom_output.json"
        assert filepath.exists()

    def test_save_empty_list(self, *, tmp_path: Path) -> None:
        """save_traces handles an empty list of traces."""
        filepath = save_traces(traces=[], output_dir=tmp_path)

        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data == []

    def test_save_traces_round_trip(
        self,
        *,
        sample_trace: AgentTrace,
        tmp_path: Path,
    ) -> None:
        """Saved traces can be deserialized back to AgentTrace objects."""
        filepath = save_traces(traces=[sample_trace], output_dir=tmp_path)
        data = json.loads(filepath.read_text(encoding="utf-8"))

        restored = [AgentTrace.model_validate(item) for item in data]
        assert len(restored) == 1
        assert restored[0].trace_id == sample_trace.trace_id
