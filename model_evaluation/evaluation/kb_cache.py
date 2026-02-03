"""KB caching for deterministic evaluation runs.

Pre-generates KB content for each question and provides a cached session
that replays the same content for both markdown and plain text runs.
"""

import json
from pathlib import Path

from model_evaluation.evaluation.schemas import QuestionRow
from model_evaluation.main_agent.kb_generator.agent import create_kb_generator_agent
from model_evaluation.main_agent.kb_generator.schemas import GeneratorOutput
from model_evaluation.main_agent.kb_generator.session import GeneratorSession


class CachedGeneratorSession:
    """A GeneratorSession replacement that returns pre-cached content.

    Duck-types the GeneratorSession interface so that the existing
    search_knowledge_base tool works without modification.
    """

    def __init__(self, *, cached_output: GeneratorOutput) -> None:
        """Initialize with pre-generated output.

        Args:
            cached_output: The GeneratorOutput to return for any query.
        """
        self._cached_output = cached_output

    def generate(
        self,
        *,
        query: str,
        include_private_info: bool,
        universe_context: str | None = None,
    ) -> GeneratorOutput:
        """Return the cached output regardless of input.

        Args:
            query: Ignored — cached content is returned.
            include_private_info: Ignored — cached content is returned.
            universe_context: Ignored — cached content is returned.

        Returns:
            The pre-cached GeneratorOutput.
        """
        return self._cached_output

    def reset(self) -> None:
        """No-op for cached sessions."""


def generate_kb_cache(
    *,
    questions: list[QuestionRow],
    output_path: Path,
) -> dict[int, GeneratorOutput]:
    """Pre-generate KB content for all questions and save to disk.

    Creates a real GeneratorSession and generates content for each question,
    resetting between questions to ensure independence.

    Args:
        questions: List of questions to generate KB content for.
        output_path: Path to save the cache JSON file.

    Returns:
        Dictionary mapping question number to GeneratorOutput.
    """
    agent, checkpointer = create_kb_generator_agent(enable_tracing=False)
    session = GeneratorSession(agent=agent, checkpointer=checkpointer)

    cache: dict[int, GeneratorOutput] = {}

    for question in questions:
        session.reset()
        print(f"  Generating KB for question {question.number}: {question.question[:60]}...")

        output = session.generate(
            query=question.question,
            include_private_info=True,
            universe_context=question.universe_context_key,
        )
        cache[question.number] = output

    save_kb_cache(cache=cache, output_path=output_path)
    return cache


def save_kb_cache(
    *,
    cache: dict[int, GeneratorOutput],
    output_path: Path,
) -> None:
    """Save KB cache to a JSON file.

    Args:
        cache: Mapping of question number to GeneratorOutput.
        output_path: Path to write the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = {str(question_id): output.model_dump() for question_id, output in cache.items()}
    output_path.write_text(
        json.dumps(serialized, indent=2),
        encoding="utf-8",
    )


def load_kb_cache(*, cache_path: Path) -> dict[int, GeneratorOutput]:
    """Load KB cache from a JSON file.

    Args:
        cache_path: Path to the JSON file.

    Returns:
        Dictionary mapping question number to GeneratorOutput.
    """
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {
        int(question_id): GeneratorOutput.model_validate(data) for question_id, data in raw.items()
    }
