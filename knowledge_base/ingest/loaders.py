from pathlib import Path

from docling.document_converter import DocumentConverter

from utils.logging import logger


def load_markdown(path: Path) -> object:
    """Load a Markdown file using Docling.

    Args:
        path: Path to the Markdown file.

    Returns:
        Docling document object.
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        return result.document
    except Exception as e:
        logger.warning(f"Failed to load markdown file {path}: {e}")
        raise


def load_pdf(path: Path) -> object:
    """Load a PDF file using Docling with OCR enabled.

    Args:
        path: Path to the PDF file.

    Returns:
        Docling document object.
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        return result.document
    except Exception as e:
        logger.warning(f"Failed to load PDF file {path}: {e}")
        raise
