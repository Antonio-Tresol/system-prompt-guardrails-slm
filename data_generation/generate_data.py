"""Main script for data generation pipeline.

This script generates synthetic data using various LLMs from OpenRouter.
It supports two modes: simple (direct generation) and agentic (with tools).
"""

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from data_generation.config import Settings
from data_generation.tools import (
    add_task,
    complete_task,
    create_critique_tool,
    get_entities,
    get_next_task,
    reset_tools,
    save_entity,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# List of allowed models
ALLOWED_MODELS = [
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5",
]

# Pool of themes for data generation
THEME_POOL = [
    "meat",
    "vegetables",
    "spices",
    "seafood",
    "dairy",
    "fruits",
    "grains",
    "beverages",
    "desserts",
    "condiments",
]


def validate_model_name(model_name: str) -> None:
    """Validate that the model name is in the allowed list.

    Args:
        model_name: The model name to validate.

    Raises:
        ValueError: If the model name is not in the allowed list.
    """
    if model_name not in ALLOWED_MODELS:
        msg = (
            f"Model '{model_name}' is not allowed. Allowed models are: {', '.join(ALLOWED_MODELS)}"
        )
        raise ValueError(msg)


def load_prompt_template(template_path: str) -> str:
    """Load a prompt template from a file.

    Args:
        template_path: Path to the prompt template file.

    Returns:
        The content of the template file.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    path = Path(template_path)
    if not path.exists():
        msg = f"Prompt template file not found: {template_path}"
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8")


def create_llm(settings: Settings, model_name: str) -> ChatOpenAI:
    """Create and configure the ChatOpenAI instance for OpenRouter.

    Args:
        settings: The settings object with API credentials.
        model_name: The model name to use.

    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=model_name,
        temperature=0.7,
    )


def generate_simple(llm: ChatOpenAI, prompt_template: str, theme: str) -> str:
    """Generate content using simple mode (no tools, no loops).

    Args:
        llm: The language model to use.
        prompt_template: The prompt template with {theme} placeholder.
        theme: The theme to fill into the template.

    Returns:
        The generated markdown content.
    """
    filled_prompt = prompt_template.replace("{theme}", theme)

    # Create a simple agent without tools
    agent = create_agent(llm, [], system_prompt=filled_prompt)

    # Run the agent
    result = agent.invoke({"messages": [HumanMessage(content="Generate the content.")]})

    # Extract content from messages
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            content = last_message.content
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)
            return str(content)
    return ""


def generate_agentic(llm: ChatOpenAI, prompt_template: str, theme: str) -> str:
    """Generate content using agentic mode (with tools and planning).

    Args:
        llm: The language model to use.
        prompt_template: The prompt template with {theme} placeholder.
        theme: The theme to fill into the template.

    Returns:
        The generated markdown content.
    """
    # Reset tool state for fresh generation
    reset_tools()

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

    # Create enhanced prompt for agentic mode
    filled_prompt = prompt_template.replace("{theme}", theme)
    system_message = f"""You are an AI agent tasked with generating high-quality content.
You have a maximum of 30 steps to complete your task, but you should be frugal with your actions.

You have access to the following tools:
- save_entity/get_entities: To maintain consistency in your content
  (e.g., character names, locations)
- add_task/get_next_task/complete_task: To plan and track your work
- critique_draft: To self-review your work before finalizing

Your task:
{filled_prompt}

Important: Your final answer must be the single, complete Markdown document.
Do not include any other text, explanations, or tool usage information in your
final answer."""

    # Create the agent using create_agent
    agent = create_agent(llm, tools, system_prompt=system_message)

    # Run the agent
    result = agent.invoke({"messages": [HumanMessage(content="Generate the content.")]})

    # Extract the final message content
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            return str(last_message.content)
    return ""


def save_output(
    content: str,
    model_name: str,
    theme: str,
    sample_num: int,
    mode: str,
    output_dir: Path,
) -> Path:
    """Save generated content to a structured directory.

    Args:
        content: The generated content to save.
        model_name: The model name used for generation.
        theme: The theme used.
        sample_num: The sample number.
        mode: The generation mode (simple or agentic).
        output_dir: The base output directory.

    Returns:
        Path to the saved file.
    """
    # Create directory structure: output_dir / model / mode / theme /
    model_safe = model_name.replace("/", "_")
    theme_dir = output_dir / model_safe / mode / theme
    theme_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{theme}_sample_{sample_num}_{timestamp}.md"
    filepath = theme_dir / filename

    # Save content
    filepath.write_text(content, encoding="utf-8")
    return filepath


def main() -> None:
    """Main entry point for the data generation script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using LLMs from OpenRouter"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="The full OpenRouter model ID",
    )
    parser.add_argument(
        "--prompt-template-path",
        required=True,
        help="Path to the prompt template file",
    )
    parser.add_argument(
        "--themes",
        type=int,
        default=3,
        help="Number of themes to randomly select (default: 3)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples to generate per theme (default: 3)",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use agentic mode with tools (default: simple mode)",
    )
    parser.add_argument(
        "--output-dir",
        default="data_generation/outputs",
        help="Base output directory (default: data_generation/outputs)",
    )

    args = parser.parse_args()

    # Validate model name
    try:
        validate_model_name(args.model_name)
    except ValueError as e:
        logger.error(str(e))
        return

    # Load settings
    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception as e:
        logger.error("Error loading settings: %s", e)
        logger.error("Make sure you have a .env file in the data_generation/ directory.")
        return

    # Load prompt template
    try:
        prompt_template = load_prompt_template(args.prompt_template_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Create LLM
    llm = create_llm(settings, args.model_name)

    # Select random themes
    if args.themes > len(THEME_POOL):
        logger.warning("Requested %d themes but only %d available.", args.themes, len(THEME_POOL))
        selected_themes = THEME_POOL
    else:
        selected_themes = random.sample(THEME_POOL, args.themes)

    logger.info("Selected themes: %s", ", ".join(selected_themes))
    logger.info("Model: %s", args.model_name)
    logger.info("Mode: %s", "agentic" if args.agentic else "simple")
    logger.info("Samples per theme: %d", args.samples)

    # Generate content
    output_dir = Path(args.output_dir)
    mode = "agentic" if args.agentic else "simple"
    total_generations = len(selected_themes) * args.samples
    current = 0

    for theme in selected_themes:
        logger.info("=" * 60)
        logger.info("Processing theme: %s", theme)
        logger.info("=" * 60)

        for sample_num in range(1, args.samples + 1):
            current += 1
            logger.info(
                "[%d/%d] Generating sample %d for %s...",
                current,
                total_generations,
                sample_num,
                theme,
            )

            try:
                if args.agentic:
                    content = generate_agentic(llm, prompt_template, theme)
                else:
                    content = generate_simple(llm, prompt_template, theme)

                # Save output
                filepath = save_output(
                    content,
                    args.model_name,
                    theme,
                    sample_num,
                    mode,
                    output_dir,
                )
                logger.info("✓ Saved to: %s", filepath)

            except Exception as e:
                logger.error("✗ Error generating sample: %s", e)
                continue

    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info("Total files generated: %d", current)
    logger.info("Output directory: %s", output_dir.absolute())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
