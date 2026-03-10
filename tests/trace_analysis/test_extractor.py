"""Tests for trace summary extraction and rendering."""

from pathlib import Path

from paper.trace_analysis.extractor import (
    TraceSummary,
    render_summary,
    summarize_trace,
)
from paper.trace_analysis.index import OutcomeType, TraceIndex


class TestSummarizeTrace:
    """Tests for the summarize_trace function."""

    def test_basic_trace_summary(self, synthetic_results_dir: Path) -> None:
        """Should produce a valid TraceSummary from a trace record."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        record = index.pairs[("4b", 1)]["markdown"]

        summary = summarize_trace(record=record)

        assert isinstance(summary, TraceSummary)
        assert summary.trace_id == record.trace_id
        assert summary.question_id == 1
        assert summary.prompt_format == "markdown"
        assert summary.model_size == "4b"

    def test_detects_tool_usage(self, synthetic_results_dir: Path) -> None:
        """Should detect which tools were used in the trace."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        # Q1 (no think tool) — regular search + response
        record_no_think = index.pairs[("4b", 1)]["markdown"]
        summary_no_think = summarize_trace(record=record_no_think)

        assert not summary_no_think.used_think_tool
        assert summary_no_think.think_positions == []
        assert "search_knowledge_base" in summary_no_think.tools_used

    def test_detects_think_tool_usage(self, synthetic_results_dir: Path) -> None:
        """Should detect when the think tool is used and at which step."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        # Q4 markdown uses think tool
        record_with_think = index.pairs[("4b", 4)]["markdown"]
        summary_with_think = summarize_trace(record=record_with_think)

        assert summary_with_think.used_think_tool
        assert 1 in summary_with_think.think_positions
        assert "think" in summary_with_think.tools_used

    def test_extracts_search_queries(self, synthetic_results_dir: Path) -> None:
        """Should extract all search queries from the trace."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        record = index.pairs[("4b", 1)]["markdown"]

        summary = summarize_trace(record=record)

        assert len(summary.search_queries) == 1
        assert summary.search_queries[0] == "What are the opening hours?"

    def test_extracts_privacy_labels(self, synthetic_results_dir: Path) -> None:
        """Should extract privacy labels from KB results."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        # Q2 (private question) should have Private labels
        record = index.pairs[("4b", 2)]["markdown"]
        summary = summarize_trace(record=record)

        assert "Private" in summary.privacy_labels_received

    def test_detects_inline_reasoning(self, synthetic_results_dir: Path) -> None:
        """Should detect when model reasons inline (in text before tool call)."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        record = index.pairs[("4b", 1)]["markdown"]

        summary = summarize_trace(record=record)

        # The synthetic trace has "I need to search for this information."
        # which starts with "I need to" — a reasoning indicator
        search_step = next(s for s in summary.steps if s.tool_calls)
        assert search_step.has_inline_reasoning

    def test_step_count_matches(self, synthetic_results_dir: Path) -> None:
        """Summary step count should match the actual steps in the trace."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        # Q1 (no think): 2 steps (search + response)
        record_no_think = index.pairs[("4b", 1)]["markdown"]
        summary_no_think = summarize_trace(record=record_no_think)
        assert summary_no_think.total_steps == 2

        # Q4 MD (with think): 3 steps (think + search + response)
        record_with_think = index.pairs[("4b", 4)]["markdown"]
        summary_with_think = summarize_trace(record=record_with_think)
        assert summary_with_think.total_steps == 3

    def test_preserves_outcome(self, synthetic_results_dir: Path) -> None:
        """Summary should preserve the outcome from the record."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        tp_record = index.pairs[("4b", 2)]["markdown"]
        summary = summarize_trace(record=tp_record)
        assert summary.outcome == OutcomeType.TP

        tn_record = index.pairs[("4b", 1)]["markdown"]
        summary = summarize_trace(record=tn_record)
        assert summary.outcome == OutcomeType.TN


class TestRenderSummary:
    """Tests for the render_summary function."""

    def test_renders_readable_output(self, synthetic_results_dir: Path) -> None:
        """Should produce human-readable text with key information."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        record = index.pairs[("4b", 1)]["markdown"]
        summary = summarize_trace(record=record)

        rendered = render_summary(summary)

        assert "Trace" in rendered
        assert "What are the opening hours?" in rendered
        assert "the_carnelian_table" in rendered
        assert "markdown" in rendered
        assert "4b" in rendered

    def test_renders_think_tool_status(self, synthetic_results_dir: Path) -> None:
        """Should clearly indicate whether think tool was used or not."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        # No think tool
        record = index.pairs[("4b", 1)]["markdown"]
        summary = summarize_trace(record=record)
        rendered = render_summary(summary)
        assert "Think tool: NOT USED" in rendered

        # With think tool
        record = index.pairs[("4b", 4)]["markdown"]
        summary = summarize_trace(record=record)
        rendered = render_summary(summary)
        assert "Think tool used at steps:" in rendered

    def test_renders_step_details(self, synthetic_results_dir: Path) -> None:
        """Should include step-by-step details with tool calls."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        record = index.pairs[("4b", 1)]["markdown"]
        summary = summarize_trace(record=record)

        rendered = render_summary(summary)

        assert "Step 1" in rendered
        assert "Step 2" in rendered
        assert "search_knowledge_base" in rendered

    def test_renders_final_answer(self, synthetic_results_dir: Path) -> None:
        """Should include the final answer text."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        record = index.pairs[("4b", 2)]["markdown"]
        summary = summarize_trace(record=record)

        rendered = render_summary(summary)

        assert "Final answer:" in rendered
        assert "cannot share this private information" in rendered
