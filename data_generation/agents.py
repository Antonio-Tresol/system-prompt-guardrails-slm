"""Agent definitions for data generation pipeline.

This module defines both simple and agentic agents with custom state management
for LangSmith tracing visibility. Global instances are provided for Studio debugging.
"""

from typing import Any

from langchain.agents import AgentState, create_agent
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import NotRequired

from data_generation.config import Settings
from data_generation.tools import (
    add_task,
    complete_task,
    create_critique_tool,
    get_entities,
    get_next_task,
    save_entity,
)


class CustomAgentState(AgentState):
    """Custom agent state with consistency store and todo list for tracking.

    Extends the base AgentState with additional fields for maintaining
    consistency and task tracking that are visible in LangSmith tracing.

    Attributes:
        consistency_store: Dictionary storing entities by category for consistency.
        todo_list: List of tasks for planning and workflow management.
        next_task_id: Counter for generating unique task IDs.
    """

    consistency_store: NotRequired[dict[str, dict[str, Any]]]
    todo_list: NotRequired[list[dict[str, Any]]]
    next_task_id: NotRequired[int]


def create_simple_agent(llm: BaseChatModel, system_prompt: str) -> CompiledStateGraph:
    """Create a simple agent without tools for direct generation.

    Args:
        llm: The language model to use.
        system_prompt: The system prompt for the agent.

    Returns:
        A configured simple agent (CompiledStateGraph from LangGraph).
    """
    agent = create_agent(llm, [], system_prompt=system_prompt)
    return agent


def create_agentic_agent(llm: BaseChatModel, system_prompt: str) -> CompiledStateGraph:
    """Create an agentic agent with tools and custom state for complex generation.

    Args:
        llm: The language model to use.
        system_prompt: The system prompt for the agent.

    Returns:
        A configured agentic agent with tools and custom state (CompiledStateGraph).
    """
    # Create the critique tool with the LLM
    critique_tool = create_critique_tool(llm)

    # Prepare tools
    tools = [
        save_entity,
        get_entities,
        add_task,
        get_next_task,
        complete_task,
        critique_tool,
    ]

    # Create the agent with custom state schema
    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        state_schema=CustomAgentState,
    )

    return agent


# Global instances for LangGraph Studio debugging
# Default model for Studio testing
DEFAULT_STUDIO_MODEL = "google/gemini-2.5-pro"


def create_studio_simple_agent() -> CompiledStateGraph:
    """Create a simple agent instance for Studio with default configuration.

    Returns:
        A simple agent configured for Studio debugging.
    """
    # Load settings (will need .env file)
    try:
        settings = Settings()  # type: ignore[call-arg]
        llm = ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=DEFAULT_STUDIO_MODEL,
            temperature=0.7,
        )
    except Exception:
        # Fallback for Studio without .env
        llm = ChatOpenAI(model=DEFAULT_STUDIO_MODEL, temperature=0.7)

    system_prompt = "You are a helpful assistant that generates high-quality content."
    return create_simple_agent(llm, system_prompt)


def create_studio_agentic_agent() -> CompiledStateGraph:
    """Create an agentic agent instance for Studio with default configuration.

    Returns:
        An agentic agent configured for Studio debugging.
    """
    # Load settings (will need .env file)
    try:
        settings = Settings()  # type: ignore[call-arg]
        llm = ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=DEFAULT_STUDIO_MODEL,
            temperature=0.7,
        )
    except Exception:
        # Fallback for Studio without .env
        llm = ChatOpenAI(model=DEFAULT_STUDIO_MODEL, temperature=0.7)

    system_prompt = """You are an AI agent tasked with generating high-quality content.
You have a maximum of 30 steps to complete your task, but you should be frugal with your actions.

You have access to the following tools:
- save_entity/get_entities: To maintain consistency in your content
  (e.g., character names, locations)
- add_task/get_next_task/complete_task: To plan and track your work
- critique_draft: To self-review your work before finalizing

Your final answer must be the single, complete Markdown document.
Do not include any other text, explanations, or tool usage information in your
final answer."""

    return create_agentic_agent(llm, system_prompt)


# Create global instances for Studio
# Note: These are lazy-loaded functions. Call them to get the agent instances.
# Example: agent = simple_agent_studio()
simple_agent_studio = create_studio_simple_agent
agentic_agent_studio = create_studio_agentic_agent
