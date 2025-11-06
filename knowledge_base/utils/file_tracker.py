import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileTracker:
    """Track ingested files and their modification times."""

    def __init__(self, tracker_file: str) -> None:
        """Initialize file tracker.

        Args:
            tracker_file: Path to the tracker JSON file.
        """
        self.tracker_file = Path(tracker_file)
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        self.tracked_files: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        """Load tracked files from JSON file."""
        if self.tracker_file.exists():
            try:
                with self.tracker_file.open() as f:
                    self.tracked_files = json.load(f)
                logger.info(f"Loaded {len(self.tracked_files)} tracked files")
            except Exception as e:
                logger.warning(f"Failed to load tracker file: {e}")
                self.tracked_files = {}
        else:
            self.tracked_files = {}

    def _save(self) -> None:
        """Save tracked files to JSON file."""
        try:
            with self.tracker_file.open("w") as f:
                json.dump(self.tracked_files, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tracker file: {e}")

    def track_file(self, path: Path, timestamp: float) -> None:
        """Track a file with its modification timestamp.

        Args:
            path: Path to the file.
            timestamp: Modification timestamp.
        """
        self.tracked_files[str(path)] = timestamp
        self._save()

    def is_file_updated(self, path: Path) -> bool:
        """Check if a file has been modified since last tracking.

        Args:
            path: Path to the file.

        Returns:
            True if file is new or modified, False otherwise.
        """
        path_str = str(path)
        if path_str not in self.tracked_files:
            return True

        try:
            current_mtime = path.stat().st_mtime
            tracked_mtime = self.tracked_files[path_str]
            return current_mtime > tracked_mtime
        except Exception as e:
            logger.warning(f"Failed to check file modification time: {e}")
            return True

    def get_unprocessed_files(self, source_dir: Path) -> list[Path]:
        """Get list of new or modified files in source directory.

        Args:
            source_dir: Directory containing source documents.

        Returns:
            List of paths to unprocessed files.
        """
        unprocessed: list[Path] = []

        for ext in ["*.md", "*.pdf"]:
            for file_path in source_dir.rglob(ext):
                if self.is_file_updated(file_path):
                    unprocessed.append(file_path)

        return unprocessed
