"""Agent definitions for data generation pipeline.

This module defines both simple and agentic agents.
- Simple agent: One-shot generation without tools
- Agentic agent: Uses Deep Agents with planning, tools, and context management
"""

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph

from data_generation.config import Settings
from data_generation.constants import (
    DEFAULT_DATA_GENERATION_PROMPT_TEMPLATE,
    DEFAULT_MODEL,
    DEFAULT_THEME,
)
from data_generation.internal_prompts import DEEP_AGENT_SYSTEM_PROMPT
from data_generation.utils import load_prompt_template

# Critique Subagent Definition
# Used by create_deep_writing_agent to provide feedback on generated content
CRITIQUE_SUBAGENT = {
    "name": "critique_draft",
    "description": (
        "Use this subagent to review and critique your draft content. "
        "It will provide feedback on completeness, consistency, tone, and quality. "
        "Send the general requirements and the draft content to be reviewed."
    ),
    "system_prompt": (
        "You are a critical reviewer. Review the provided content and give "
        "constructive feedback.\n\n"
        "Check for:\n"
        "1. Completeness: Is all required information present?\n"
        "2. Consistency: Are there any contradictions or inconsistencies?\n"
        "3. Tone: Is the tone appropriate for the content?\n"
        "4. Quality: Is the content well-written and engaging?\n\n"
        "Provide specific, actionable feedback that the writer can use to improve "
        "their work."
    ),
    "tools": [],
}


def create_one_shot_writing_agent(model: BaseChatModel, system_prompt: str) -> CompiledStateGraph:
    """Create a simple agent without tools for direct generation.

    Args:
        model: The  model to use.
        system_prompt: The system prompt for the agent.

    Returns:
        A configured simple agent (CompiledStateGraph from LangGraph).
    """
    agent = create_agent(model=model, system_prompt=system_prompt, name="one_shot_writing_agent")
    return agent


def create_deep_writing_agent(model: BaseChatModel, system_prompt: str) -> CompiledStateGraph:
    """Create a Deep Agent with planning, tools, and context management.

    This uses the deepagents library to create an agent with:
    - Built-in planning (TodoListMiddleware)
    - File system for context management (FilesystemMiddleware)
    - Critique subagent for self-assessment

    Args:
        model: The model to use.
        system_prompt: The system prompt for the agent.

    Returns:
        A configured Deep Agent (CompiledStateGraph).
    """
    # Create a Deep Agent with built-in planning and filesystem tools
    # The agent automatically gets:
    # - write_todos tool (planning)
    # - ls, read_file, write_file, edit_file (filesystem)
    # Plus our critique subagent
    agent = create_deep_agent(
        model=model,
        subagents=[CRITIQUE_SUBAGENT],  # type: ignore[arg-type]
        system_prompt=system_prompt,
        name="deep_writing_agent",
    )

    return agent


# Global instances for LangGraph Studio debugging
# Default model for Studio testing
def _create_studio_model_and_tracer() -> tuple[ChatOpenAI, CallbackHandler]:
    """Create a model instance with Langfuse tracing for Studio.

    Returns:
        Tuple of (model, langfuse_handler) configured for tracing.
    """
    settings = Settings()  # type: ignore[call-arg]
    model = ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=DEFAULT_MODEL,
        temperature=0.7,
    )
    # Initialize Langfuse for tracing
    _langfuse_client = Langfuse(  # noqa: F841
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_base_url,
    )
    langfuse_handler = CallbackHandler(public_key=settings.langfuse_public_key)
    return model, langfuse_handler


def create_one_shot_writing_agent_for_studio() -> CompiledStateGraph:
    """Create a simple agent instance for Studio with default configuration.

    Returns:
        A simple agent configured for Studio debugging with Langfuse tracing.
    """
    model, langfuse_handler = _create_studio_model_and_tracer()
    system_prompt = load_prompt_template(
        DEFAULT_DATA_GENERATION_PROMPT_TEMPLATE,
        theme=DEFAULT_THEME,
    )
    agent = create_one_shot_writing_agent(model, system_prompt)
    return agent.with_config(callbacks=[langfuse_handler])


def create_deep_writing_agent_for_studio() -> CompiledStateGraph:
    """Create a Deep Agent instance for Studio with default configuration.

    For interactive testing in Studio, you should send the corpus generation
    requirements as a user message when you start the chat. The agent is
    configured with system instructions only.

    Returns:
        A Deep Agent configured for Studio debugging with Langfuse tracing.
    """
    model, langfuse_handler = _create_studio_model_and_tracer()
    # Use the system prompt as pure instructions, not embedded with corpus task
    agent = create_deep_writing_agent(model, DEEP_AGENT_SYSTEM_PROMPT)
    return agent.with_config(callbacks=[langfuse_handler])
