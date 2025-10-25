"""Custom tools for the agentic data generation mode with state management.

This module provides tools that allow the agent to maintain consistency,
plan its work, and self-critique its outputs. Tools update the agent
state directly for visibility in LangSmith tracing using Command.
"""

from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command


@tool
def save_entity(  # noqa: D417
    category: str,
    name: str,
    data: str,
    runtime: ToolRuntime,
) -> Command:
    """Save an entity to the consistency store for later retrieval.

    Use this tool to remember important details like character names, currency,
    locations, or any other fictional elements that should remain consistent
    throughout the generated document.

    Args:
        category: The category of the entity (e.g., "staff", "currency", "location").
        name: The unique name or identifier of the entity.
        data: The data or description to store for this entity.

    Returns:
        Command to update state with the saved entity.
    """
    # Get current consistency store from state
    consistency_store = runtime.state.get("consistency_store", {})

    # Update the consistency store
    if category not in consistency_store:
        consistency_store[category] = {}
    consistency_store[category][name] = data

    # Return command to update state
    return Command(
        update={
            "consistency_store": consistency_store,
            "messages": [
                ToolMessage(
                    f"Saved {name} in category {category}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def get_entities(  # noqa: D417
    category: str,
    runtime: ToolRuntime,
) -> str:
    """Retrieve all entities from a specific category.

    Use this tool to recall previously saved entities and maintain consistency
    in your generated content.

    Args:
        category: The category to retrieve entities from.

    Returns:
        A string representation of all entities in that category.
    """
    consistency_store = runtime.state.get("consistency_store", {})

    if category not in consistency_store:
        return f"No entities found in category {category}."
    entities = consistency_store[category]
    if not entities:
        return f"No entities found in category {category}."
    return "\n".join([f"{name}: {data}" for name, data in entities.items()])


@tool
def add_task(  # noqa: D417
    task_description: str,
    runtime: ToolRuntime,
) -> Command:
    """Add a new task to the todo list.

    Use this tool to break down the document generation into manageable steps
    and track what needs to be done.

    Args:
        task_description: Description of the task to add.

    Returns:
        Command to update state with the new task.
    """
    # Get current todo list and task ID from state
    todo_list = runtime.state.get("todo_list", [])
    next_task_id = runtime.state.get("next_task_id", 1)

    # Add new task
    task_id = next_task_id
    todo_list.append(
        {
            "id": task_id,
            "description": task_description,
            "completed": False,
        }
    )

    # Return command to update state
    return Command(
        update={
            "todo_list": todo_list,
            "next_task_id": next_task_id + 1,
            "messages": [
                ToolMessage(
                    f"Task {task_id} added: {task_description}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def get_next_task(  # noqa: D417
    runtime: ToolRuntime,
) -> str:
    """Get the next incomplete task from the todo list.

    Use this tool to see what you should work on next.

    Returns:
        The next incomplete task or a message if all tasks are complete.
    """
    todo_list = runtime.state.get("todo_list", [])

    for task in todo_list:
        if not task["completed"]:
            return f"Task {task['id']}: {task['description']}"
    return "All tasks completed!"


@tool
def complete_task(  # noqa: D417
    task_id: int,
    runtime: ToolRuntime,
) -> Command:
    """Mark a task as completed.

    Use this tool when you finish a task to track your progress.

    Args:
        task_id: The ID of the task to mark as complete.

    Returns:
        Command to update state with the completed task.
    """
    todo_list = runtime.state.get("todo_list", [])

    # Find and update the task
    task_found = False
    for task in todo_list:
        if task["id"] == task_id:
            task["completed"] = True
            task_found = True
            break

    if not task_found:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Task {task_id} not found.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    # Return command to update state
    return Command(
        update={
            "todo_list": todo_list,
            "messages": [
                ToolMessage(
                    f"Task {task_id} marked as complete.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


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
