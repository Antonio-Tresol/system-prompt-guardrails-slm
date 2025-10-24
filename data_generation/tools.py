"""Custom tools for the agentic data generation mode.

This module provides tools that allow the agent to maintain consistency,
plan its work, and self-critique its outputs.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

# In-memory storage for the InternalConsistencyTool
_consistency_store: dict[str, dict[str, Any]] = {}


@tool
def save_entity(category: str, name: str, data: str) -> str:
    """Save an entity to the consistency store for later retrieval.

    Use this tool to remember important details like character names, currency,
    locations, or any other fictional elements that should remain consistent
    throughout the generated document.

    Args:
        category: The category of the entity (e.g., "staff", "currency", "location").
        name: The unique name or identifier of the entity.
        data: The data or description to store for this entity.

    Returns:
        Confirmation message that the entity was saved.
    """
    if category not in _consistency_store:
        _consistency_store[category] = {}
    _consistency_store[category][name] = data
    return f"Saved {name} in category {category}."


@tool
def get_entities(category: str) -> str:
    """Retrieve all entities from a specific category.

    Use this tool to recall previously saved entities and maintain consistency
    in your generated content.

    Args:
        category: The category to retrieve entities from.

    Returns:
        A string representation of all entities in that category.
    """
    if category not in _consistency_store:
        return f"No entities found in category {category}."
    entities = _consistency_store[category]
    if not entities:
        return f"No entities found in category {category}."
    return "\n".join([f"{name}: {data}" for name, data in entities.items()])


# In-memory storage for the TodoListTool
_todo_list: list[dict[str, Any]] = []
_next_task_id: int = 1


@tool
def add_task(task_description: str) -> str:
    """Add a new task to the todo list.

    Use this tool to break down the document generation into manageable steps
    and track what needs to be done.

    Args:
        task_description: Description of the task to add.

    Returns:
        Confirmation message with the task ID.
    """
    global _next_task_id
    task_id = _next_task_id
    _next_task_id += 1
    _todo_list.append({"id": task_id, "description": task_description, "completed": False})
    return f"Task {task_id} added: {task_description}"


@tool
def get_next_task() -> str:
    """Get the next incomplete task from the todo list.

    Use this tool to see what you should work on next.

    Returns:
        The next incomplete task or a message if all tasks are complete.
    """
    for task in _todo_list:
        if not task["completed"]:
            return f"Task {task['id']}: {task['description']}"
    return "All tasks completed!"


@tool
def complete_task(task_id: int) -> str:
    """Mark a task as completed.

    Use this tool when you finish a task to track your progress.

    Args:
        task_id: The ID of the task to mark as complete.

    Returns:
        Confirmation message or error if task not found.
    """
    for task in _todo_list:
        if task["id"] == task_id:
            task["completed"] = True
            return f"Task {task_id} marked as complete."
    return f"Task {task_id} not found."


def create_critique_tool(llm: BaseChatModel) -> BaseTool:
    """Create a critique tool that uses the provided LLM for self-assessment.

    Args:
        llm: The language model to use for critique.

    Returns:
        A tool that can critique draft markdown.
    """

    @tool
    def critique_draft(draft_markdown: str) -> str:
        """Critique a draft markdown document for compliance, tone, and completeness.

        Use this tool to get feedback on your draft before finalizing it.
        The critique will check for:
        - Compliance with requirements
        - Appropriate tone and style
        - Completeness of content

        Args:
            draft_markdown: The draft markdown document to critique.

        Returns:
            Feedback and suggestions for improvement.
        """
        critique_prompt = f"""You are a critical reviewer. Review the following markdown \
document and provide constructive feedback.

Check for:
1. Completeness: Is all required information present?
2. Consistency: Are there any contradictions or inconsistencies?
3. Tone: Is the tone appropriate for the content?
4. Quality: Is the content well-written and engaging?

Provide specific, actionable feedback.

Document to review:
{draft_markdown}

Your critique:"""

        response = llm.invoke(critique_prompt)
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)
            return str(content)
        return str(response)

    return critique_draft


def reset_tools() -> None:
    """Reset all tool state (useful for testing or between generations)."""
    global _consistency_store, _todo_list, _next_task_id
    _consistency_store = {}
    _todo_list = []
    _next_task_id = 1
