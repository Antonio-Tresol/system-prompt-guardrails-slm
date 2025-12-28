"""Script for downloading specific arXiv papers as PDFs.

This utility script creates a directory and downloads
a predefined set of arXiv papers in PDF format.
"""

import logging
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAVE_DIR = Path("papers")
ARXIV_URLS = [
    "https://arxiv.org/pdf/2510.18234",
    "https://arxiv.org/pdf/2510.18871",
    "https://arxiv.org/pdf/2510.19818",
    "https://arxiv.org/pdf/2510.20579",
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def download_file(url: str, dest_dir: Path) -> None:
    """Download a single PDF file from arXiv.

    Args:
        url: The URL of the arXiv PDF file.
        dest_dir: The directory where the file will be saved.

    Raises:
        requests.exceptions.RequestException: If the request fails.
    """
    filename = dest_dir / f"{url.split('/')[-1]}.pdf"
    logger.info("Downloading %s -> %s", url, filename)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(filename, "wb") as f:
        f.write(response.content)

    logger.info("✓ Successfully downloaded: %s", filename.name)


def download_all_papers(urls: list[str], dest_dir: Path) -> None:
    """Download all arXiv papers to the specified directory.

    Args:
        urls: List of arXiv PDF URLs.
        dest_dir: Directory where PDFs will be saved.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created directory: %s", dest_dir.resolve())

    for url in urls:
        try:
            download_file(url, dest_dir)
        except requests.exceptions.RequestException as e:
            logger.error("✗ Failed to download %s: %s", url, e)

    logger.info("All downloads completed. ✅")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the arXiv paper downloader."""
    download_all_papers(ARXIV_URLS, SAVE_DIR)


if __name__ == "__main__":
    main()
