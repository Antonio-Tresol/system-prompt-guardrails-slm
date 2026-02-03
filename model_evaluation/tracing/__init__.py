"""Local trajectory tracing for the safety evaluation pipeline.

Provides middleware-based trace capture for the main RAG agent,
producing structured JSON traces without external service dependencies.
"""

from model_evaluation.tracing.middleware import TrajectoryCapture
from model_evaluation.tracing.schemas import AgentStep, AgentTrace, ToolCallTrace
from model_evaluation.tracing.storage import save_trace, save_traces

__all__ = [
    "AgentStep",
    "AgentTrace",
    "ToolCallTrace",
    "TrajectoryCapture",
    "save_trace",
    "save_traces",
]
