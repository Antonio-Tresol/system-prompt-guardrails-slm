"""Main script for data generation pipeline.

This script generates synthetic data using various LLMs from OpenRouter.
It supports two modes: simple (direct generation) and agentic (with tools).
"""

import argparse
import logging
import random
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from data_generation.agents import (
    create_deep_writing_agent,
    create_one_shot_writing_agent,
)
from data_generation.config import Settings
from data_generation.constants import (
    ALLOWED_MODELS,
    DEFAULT_DATA_GENERATION_PROMPT_TEMPLATE,
    DEFAULT_MODEL,
    THEME_POOL,
)
from data_generation.internal_prompts import DEEP_AGENT_SYSTEM_PROMPT
from data_generation.utils import load_prompt_template, save_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def create_model(settings: Settings, model_name: str) -> ChatOpenAI:
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


def create_langfuse_handler(settings: Settings) -> CallbackHandler:
    """Create and configure the Langfuse callback handler.

    Args:
        settings: The settings object with Langfuse credentials.

    Returns:
        Configured Langfuse CallbackHandler instance.
    """
    # Create Langfuse client with explicit credentials
    # This initializes the global Langfuse instance that CallbackHandler will use
    _langfuse_client = Langfuse(  # noqa: F841
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_base_url,
    )
    # Create callback handler (it will use the global langfuse client automatically)
    return CallbackHandler()


def generate_content_with_one_shot_agent(
    model: ChatOpenAI, prompt_template: str, theme: str, langfuse_handler: CallbackHandler
) -> str:
    """Generate content using one-shot writing (no tools, no loops).

    Args:
        model: The  model to use.
        prompt_template: The prompt template with {theme} placeholder.
        theme: The theme to fill into the template.
        langfuse_handler: The Langfuse callback handler for tracing.

    Returns:
        The generated markdown content.
    """
    filled_prompt = prompt_template.replace("{theme}", theme)
    agent = create_one_shot_writing_agent(model, filled_prompt)

    # Run the agent
    result = agent.invoke(
        {"messages": [HumanMessage(content="Generate the content.")]}, callbacks=[langfuse_handler]
    )

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


def generate_content_with_deep_agent(
    model: ChatOpenAI, prompt_template: str, theme: str, langfuse_handler: CallbackHandler
) -> str:
    """Generate content using Deep Agent (with planning, tools, and context management).

    Args:
        model: The model to use.
        prompt_template: The prompt template with {theme} placeholder.
        theme: The theme to fill into the template.
        langfuse_handler: The Langfuse callback handler for tracing.

    Returns:
        The generated markdown content.
    """
    # Fill in the theme placeholder - this is the actual task specification
    filled_prompt = prompt_template.replace("{theme}", theme)

    # Create agent with the system instructions (not the task)
    agent = create_deep_writing_agent(model, DEEP_AGENT_SYSTEM_PROMPT)

    # Invoke the agent with the corpus generation requirements as the user message
    result = agent.invoke(
        {"messages": [HumanMessage(content=filled_prompt)]},
        config={"callbacks": [langfuse_handler]},
    )

    # Extract the final message content
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            return str(last_message.content)
    return ""


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using LLMs from OpenRouter"
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="The full OpenRouter model ID (default: z-ai/glm-4.6)",
    )
    parser.add_argument(
        "--prompt-template-path",
        default=str(DEFAULT_DATA_GENERATION_PROMPT_TEMPLATE),
        help=f"Path to data generation prompt (default: {DEFAULT_DATA_GENERATION_PROMPT_TEMPLATE})",
    )
    parser.add_argument(
        "--themes",
        type=int,
        default=1,
        help="Number of themes to randomly select (default: 1)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples to generate per theme (default: 1)",
    )
    parser.add_argument(
        "--deep-agent",
        action="store_true",
        help="Use Deep Agent mode with planning and tools (default: simple one-shot mode)",
    )
    parser.add_argument(
        "--output-dir",
        default="data_generation/outputs",
        help="Base output directory (default: data_generation/outputs)",
    )
    return parser.parse_args()


def setup_generation(
    args: argparse.Namespace,
) -> tuple[Settings, ChatOpenAI, str, CallbackHandler]:
    """Set up the generation environment.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (settings, llm, prompt_template, langfuse_handler).

    Raises:
        SystemExit: If setup fails.
    """
    try:
        validate_model_name(args.model_name)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1) from e

    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception as e:
        logger.error("Error loading settings: %s", e)
        logger.error("Make sure you have a .env file in the data_generation/ directory.")
        raise SystemExit(1) from e

    try:
        prompt_template = load_prompt_template(args.prompt_template_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1) from e

    model = create_model(settings, args.model_name)
    langfuse_handler = create_langfuse_handler(settings)

    return settings, model, prompt_template, langfuse_handler


def select_themes(num_themes: int) -> list[str]:
    """Select random themes from the theme pool.

    Args:
        num_themes: Number of themes to select.

    Returns:
        List of selected themes.
    """
    if num_themes > len(THEME_POOL):
        logger.warning("Requested %d themes but only %d available.", num_themes, len(THEME_POOL))
        return THEME_POOL
    return random.sample(THEME_POOL, num_themes)


def run_generation(
    model: ChatOpenAI,
    prompt_template: str,
    selected_themes: list[str],
    args: argparse.Namespace,
    langfuse_handler: CallbackHandler,
) -> int:
    """Run the data generation process.

    Args:
        model: The  model to use.
        prompt_template: The prompt template.
        selected_themes: List of themes to process.
        args: Parsed command-line arguments.
        langfuse_handler: The Langfuse callback handler for tracing.

    Returns:
        Number of files generated.
    """
    output_dir = Path(args.output_dir)
    mode = "deep-agent" if args.deep_agent else "simple"
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
                if args.deep_agent:
                    content = generate_content_with_deep_agent(
                        model, prompt_template, theme, langfuse_handler
                    )
                else:
                    content = generate_content_with_one_shot_agent(
                        model, prompt_template, theme, langfuse_handler
                    )

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

    return current


def main() -> None:
    """Main entry point for the data generation script."""
    args = parse_arguments()

    settings, model, prompt_template, langfuse_handler = setup_generation(args)

    selected_themes = select_themes(args.themes)

    logger.info("Selected themes: %s", ", ".join(selected_themes))
    logger.info("Model: %s", args.model_name)
    logger.info("Mode: %s", "deep-agent" if args.deep_agent else "simple")
    logger.info("Samples per theme: %d", args.samples)

    files_generated = run_generation(
        model, prompt_template, selected_themes, args, langfuse_handler
    )

    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info("Total files generated: %d", files_generated)
    logger.info("Output directory: %s", Path(args.output_dir).absolute())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
