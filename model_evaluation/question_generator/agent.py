"""Question generator agent using Deep Agents.

This module creates a Deep Agent for generating synthetic test questions
with access to universe contexts and planning capabilities.
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, SecretStr

from model_evaluation.config import Settings
from model_evaluation.question_generator.prompts import (
    QUESTION_GENERATOR_SYSTEM_PROMPT,
    UNIVERSE_TASK_TEMPLATE,
)

# Path to universe context YAML files
UNIVERSE_CONTEXTS_DIR = (
    Path(__file__).parent.parent.parent / "data_generation" / "universe_contexts"
)


class GeneratedQuestion(BaseModel):
    """A single generated question."""

    question: str = Field(description="The question text")
    universe_context: str = Field(description="Universe context name (e.g., 'moonlit_granary')")
    is_refusal: bool = Field(description="True if question should be refused")


class QuestionGeneratorOutput(BaseModel):
    """Structured output from the question generator."""

    questions: list[GeneratedQuestion] = Field(
        description="List of generated questions",
        min_length=1,
    )


@tool
def think(thought: str) -> str:
    """Use this tool to think out loud, plan, and reason through the task.

    Args:
        thought: Your reasoning, analysis, or plan.
    """
    return f"Thought recorded: {thought}"


def load_universe_contexts() -> dict[str, str]:
    """Load all universe context YAML files.

    Returns:
        Dictionary mapping universe name to YAML content.
    """
    contexts = {}
    for yaml_file in UNIVERSE_CONTEXTS_DIR.glob("*.yaml"):
        universe_name = yaml_file.stem
        contexts[universe_name] = yaml_file.read_text(encoding="utf-8")
    return contexts


def format_universe_contexts_for_prompt(contexts: dict[str, str]) -> str:
    """Format universe contexts for inclusion in the prompt.

    Args:
        contexts: Dictionary of universe name to YAML content.

    Returns:
        Formatted string with all contexts.
    """
    parts = []
    for name, content in contexts.items():
        parts.append(f"### {name}\n\n```yaml\n{content}\n```\n")
    return "\n".join(parts)


def _create_langfuse_handler(settings: Settings) -> CallbackHandler:
    """Create a Langfuse callback handler for tracing.

    Args:
        settings: Settings object with Langfuse credentials.

    Returns:
        Configured Langfuse CallbackHandler.
    """
    _langfuse_client = Langfuse(  # noqa: F841
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_base_url,
    )
    return CallbackHandler(public_key=settings.langfuse_public_key)


def create_question_generator_agent(
    *,
    settings: Settings | None = None,
    enable_tracing: bool = True,
) -> CompiledStateGraph:
    """Create a question generator agent.

    Args:
        settings: Optional settings, loaded from env if not provided.
        enable_tracing: Whether to enable Langfuse tracing (default: True).

    Returns:
        Compiled agent for question generation.
    """
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]

    model = ChatOpenAI(
        api_key=SecretStr(settings.openrouter_api_key),
        base_url=settings.openrouter_base_url,
        model=settings.kb_generator_model,  # Use same model as KB generator
        temperature=0.8,
    )

    agent = create_agent(
        model=model,
        tools=[think],
        system_prompt=QUESTION_GENERATOR_SYSTEM_PROMPT,
        response_format=ToolStrategy(QuestionGeneratorOutput),
        name="question_generator_agent",
    )

    if enable_tracing:
        langfuse_handler = _create_langfuse_handler(settings)
        agent = agent.with_config(callbacks=[langfuse_handler])

    return agent


def generate_questions(
    *,
    num_refusal: int = 30,
    num_non_refusal: int = 30,
    existing_questions: list[str] | None = None,
    settings: Settings | None = None,
) -> list[GeneratedQuestion]:
    """Generate synthetic questions using the agent.

    Args:
        num_refusal: Number of refusal questions to generate (default: 30).
        num_non_refusal: Number of non-refusal questions to generate (default: 30).
        existing_questions: List of existing question texts to avoid duplicates.
        settings: Optional settings.

    Returns:
        List of generated questions.
    """
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]

    # Load universe contexts
    contexts = load_universe_contexts()
    formatted_contexts = format_universe_contexts_for_prompt(contexts)

    # Format existing questions
    existing_str = (
        "None" if not existing_questions else "\n".join(f"- {q}" for q in existing_questions)
    )

    # Build the task message
    task_message = UNIVERSE_TASK_TEMPLATE.format(
        num_questions=num_refusal + num_non_refusal,
        num_refusal=num_refusal,
        num_non_refusal=num_non_refusal,
        universe_contexts=formatted_contexts,
        existing_questions=existing_str,
    )

    # Create and run agent
    agent = create_question_generator_agent(settings=settings)
    result = agent.invoke({"messages": [{"role": "user", "content": task_message}]})

    return result["structured_response"].questions
