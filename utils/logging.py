"""Centralized logging configuration using Loguru for the entire project."""

import logging
import sys

from loguru import logger

# Silence verbose HTTP request logging from OpenAI/LangChain dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def setup_logging(*, level: str = "INFO", log_file: str | None = None) -> None:
    """Configure Loguru logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file name. If None, uses a default based on module.
    """
    # Remove default handler
    logger.remove()

    # Add colored console handler with custom format
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=level.upper(),
        colorize=True,
    )

    # Add file handler with rotation if log_file is specified
    if log_file:
        logger.add(
            log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            ),
            level=level.upper(),
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )


# Export logger for convenience
__all__ = ["logger", "setup_logging"]
