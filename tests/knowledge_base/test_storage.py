"""Unit tests for vector database storage and file tracking."""

from pathlib import Path

from knowledge_base.utils.file_tracker import FileTracker

# Test constants
TEST_CONTENT_ORIGINAL = "test content"
TEST_CONTENT_1 = "content 1"
TEST_CONTENT_2 = "content 2"
TEST_CONTENT_3 = "content 3"
TEST_FILE_MD = "doc1.md"
TEST_FILE_PDF = "doc2.pdf"
TEST_FILE_TXT = "doc3.txt"
MTIME_OFFSET_SECONDS = 100


class TestFileTracker:
    """Test file tracking functionality for incremental updates."""

    def test_track_and_check_file(self, tmp_path: Path) -> None:
        """Test tracking a file and checking if it needs reprocessing.

        Validates that new files are detected as updated, and tracked
        files are not flagged for reprocessing.
        """
        # Arrange: Create tracker and test file
        tracker_file = tmp_path / "tracker.json"
        tracker = FileTracker(str(tracker_file))

        test_file = tmp_path / "test.md"
        test_file.write_text(TEST_CONTENT_ORIGINAL)
        initial_mtime = test_file.stat().st_mtime

        # Act & Assert: File should be flagged as updated (not yet tracked)
        assert tracker.is_file_updated(test_file) is True

        # Act: Track the file
        tracker.track_file(path=test_file, timestamp=initial_mtime)

        # Assert: File should no longer be flagged as updated
        assert tracker.is_file_updated(test_file) is False

    def test_get_unprocessed_files(self, tmp_path: Path) -> None:
        """Test retrieving unprocessed files from directory.

        Validates that only supported file types (.md, .pdf) are returned
        and other file types are ignored.
        """
        # Arrange: Create tracker and test files
        tracker_file = tmp_path / "tracker.json"
        tracker = FileTracker(str(tracker_file))

        source_dir = tmp_path / "docs"
        source_dir.mkdir()

        (source_dir / TEST_FILE_MD).write_text(TEST_CONTENT_1)
        (source_dir / TEST_FILE_PDF).write_text(TEST_CONTENT_2)
        (source_dir / TEST_FILE_TXT).write_text(TEST_CONTENT_3)

        # Act: Get unprocessed files
        unprocessed = tracker.get_unprocessed_files(source_dir)

        # Assert: Only .md and .pdf files should be returned
        assert len(unprocessed) == 2, "Should return exactly 2 supported files"
        assert any(p.name == TEST_FILE_MD for p in unprocessed), "Should include .md file"
        assert any(p.name == TEST_FILE_PDF for p in unprocessed), "Should include .pdf file"
        assert not any(p.name == TEST_FILE_TXT for p in unprocessed), "Should exclude .txt file"

    def test_tracker_persistence(self, tmp_path: Path) -> None:
        """Test that file tracker persists data across multiple instances.

        Validates that tracked files remain tracked when FileTracker
        is recreated from the same JSON file.
        """
        # Arrange: Create test file and first tracker instance
        tracker_file = tmp_path / "tracker.json"
        test_file = tmp_path / "test.md"
        test_file.write_text("test")
        mtime = test_file.stat().st_mtime

        # Act: Track file with first instance
        tracker1 = FileTracker(str(tracker_file))
        tracker1.track_file(path=test_file, timestamp=mtime)

        # Act: Create second tracker instance from same file
        tracker2 = FileTracker(str(tracker_file))

        # Assert: Second instance should remember tracked file
        assert tracker2.is_file_updated(test_file) is False, (
            "Tracked file should persist across instances"
        )

    def test_is_file_updated_with_modification(self, tmp_path: Path) -> None:
        """Test detection of file modifications via timestamp comparison.

        Validates that files with newer modification times than tracked
        times are correctly flagged for reprocessing.
        """
        # Arrange: Create tracker and test file
        tracker_file = tmp_path / "tracker.json"
        tracker = FileTracker(str(tracker_file))

        test_file = tmp_path / "test.md"
        test_file.write_text("original content")

        # Arrange: Track file with old modification time
        old_mtime = test_file.stat().st_mtime - MTIME_OFFSET_SECONDS
        tracker.track_file(path=test_file, timestamp=old_mtime)

        # Act & Assert: File should be detected as updated
        assert tracker.is_file_updated(test_file) is True, (
            "File with newer mtime should be flagged as updated"
        )
