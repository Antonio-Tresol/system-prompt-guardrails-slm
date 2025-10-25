"""Utility functions for data generation pipeline."""

from datetime import datetime
from pathlib import Path


def load_prompt_template(template_path: Path | str, theme: str | None = None) -> str:
    """Load a prompt template from a file and optionally fill it with a theme.

    Args:
        template_path: Path to the prompt template file.
        theme: Optional theme to replace {theme} placeholder in the template.

    Returns:
        The content of the template file, with theme filled in if provided.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    path = Path(template_path)
    if not path.exists():
        msg = f"Prompt template file not found: {template_path}"
        raise FileNotFoundError(msg)

    content = path.read_text(encoding="utf-8")

    if theme is not None:
        content = content.replace("{theme}", theme)

    return content


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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{theme}_sample_{sample_num}_{timestamp}.md"
    filepath = theme_dir / filename

    filepath.write_text(content, encoding="utf-8")
    return filepath
