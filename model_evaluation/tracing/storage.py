"""Trace storage utilities for saving agent traces to JSON files."""

import json
from datetime import datetime, timezone
from pathlib import Path

from model_evaluation.tracing.schemas import AgentTrace

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "evaluation_results"


def save_trace(
    *,
    trace: AgentTrace,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save a single trace as a JSON file.

    The filename is derived from the trace ID.

    Args:
        trace: The AgentTrace to save.
        output_dir: Directory to save the file in.

    Returns:
        Path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"trace_{trace.trace_id}.json"
    filepath.write_text(
        trace.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return filepath


def save_traces(
    *,
    traces: list[AgentTrace],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    filename: str | None = None,
) -> Path:
    """Save multiple traces as a single JSON array file.

    Args:
        traces: List of AgentTrace objects to save.
        output_dir: Directory to save the file in.
        filename: Optional filename. Defaults to a timestamped name.

    Returns:
        Path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"traces_{ts}.json"

    filepath = output_dir / filename
    data = [trace.model_dump(mode="json") for trace in traces]
    filepath.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return filepath
