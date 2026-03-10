"""Session management for the KB Generator Agent.

This module provides a session manager that maintains conversation history
across multiple queries within a single evaluation run.
"""

import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from model_evaluation.main_agent.kb_generator.schemas import (
    GeneratorContext,
    GeneratorOutput,
)

# Path to universe context YAML files
UNIVERSE_CONTEXTS_DIR = Path(__file__).parent.parent.parent.parent / "data_generation" / "universes"


class GeneratorSession:
    """Manages a generator agent session for a single evaluation run.

    The session maintains conversation history via the checkpointer's thread_id.
    Each call to generate() adds to the message history, allowing the agent to
    remember what it has generated and maintain coherence.

    The agent sees its full message history and can use the think tool to
    review previous generations before creating new documents.
    """

    def __init__(
        self,
        *,
        agent: CompiledStateGraph,
        checkpointer: InMemorySaver,
    ) -> None:
        """Initialize a new generator session.

        Args:
            agent: The compiled generator agent.
            checkpointer: The memory checkpointer for state persistence.
        """
        self._agent = agent
        self._checkpointer = checkpointer
        self._thread_id = str(uuid.uuid4())

    @property
    def thread_id(self) -> str:
        """Get the current thread ID."""
        return self._thread_id

    def reset(self) -> None:
        """Reset the session with a new thread_id, clearing history."""
        self._thread_id = str(uuid.uuid4())

    def _load_universe_yaml(self, universe_context: str) -> str | None:
        """Load the YAML content for a universe context.

        Args:
            universe_context: Name of the universe (e.g., 'moonlit_granary').

        Returns:
            The YAML file content, or None if not found.
        """
        yaml_path = UNIVERSE_CONTEXTS_DIR / f"{universe_context}.yaml"
        if yaml_path.exists():
            return yaml_path.read_text(encoding="utf-8")
        return None

    def generate(
        self,
        *,
        query: str,
        include_private_info: bool,
        universe_context: str | None = None,
    ) -> GeneratorOutput:
        """Generate document chunks for a query.

        The agent maintains history across calls via the checkpointer,
        so it remembers what was previously generated and maintains coherence.
        The think tool allows the agent to review and plan before generating.

        Args:
            query: The search query from the main agent.
            include_private_info: Whether to generate private or public docs.
            universe_context: Optional universe context name for coherent generation.

        Returns:
            GeneratorOutput with 1-3 coherent document chunks.
        """
        universe_yaml = None
        if universe_context:
            universe_yaml = self._load_universe_yaml(universe_context)

        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": self._thread_id}},
            context=GeneratorContext(
                include_private_info=include_private_info,
                universe_context=universe_context,
                universe_yaml=universe_yaml,
            ),
        )
        return result["structured_response"]
