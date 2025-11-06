"""Unit tests for vector database storage."""

from pathlib import Path

from knowledge_base.utils.file_tracker import FileTracker


class TestFileTracker:
    """Test file tracking functionality."""

    def test_track_and_check_file(self, tmp_path: Path) -> None:
        """Test tracking a file and checking if it's updated."""
        tracker_file = tmp_path / "tracker.json"
        tracker = FileTracker(str(tracker_file))

        test_file = tmp_path / "test.md"
        test_file.write_text("test content")

        initial_mtime = test_file.stat().st_mtime

        assert tracker.is_file_updated(test_file) is True

        tracker.track_file(test_file, initial_mtime)

        assert tracker.is_file_updated(test_file) is False

    def test_get_unprocessed_files(self, tmp_path: Path) -> None:
        """Test getting unprocessed files from directory."""
        tracker_file = tmp_path / "tracker.json"
        tracker = FileTracker(str(tracker_file))

        source_dir = tmp_path / "docs"
        source_dir.mkdir()

        (source_dir / "doc1.md").write_text("content 1")
        (source_dir / "doc2.pdf").write_text("content 2")
        (source_dir / "doc3.txt").write_text("content 3")

        unprocessed = tracker.get_unprocessed_files(source_dir)

        assert len(unprocessed) == 2
        assert any(p.name == "doc1.md" for p in unprocessed)
        assert any(p.name == "doc2.pdf" for p in unprocessed)
        assert not any(p.name == "doc3.txt" for p in unprocessed)

    def test_tracker_persistence(self, tmp_path: Path) -> None:
        """Test that tracker persists data across instances."""
        tracker_file = tmp_path / "tracker.json"

        test_file = tmp_path / "test.md"
        test_file.write_text("test")
        mtime = test_file.stat().st_mtime

        tracker1 = FileTracker(str(tracker_file))
        tracker1.track_file(test_file, mtime)

        tracker2 = FileTracker(str(tracker_file))
        assert tracker2.is_file_updated(test_file) is False

    def test_is_file_updated_with_modification(self, tmp_path: Path) -> None:
        """Test detecting file modifications."""
        tracker_file = tmp_path / "tracker.json"
        tracker = FileTracker(str(tracker_file))

        test_file = tmp_path / "test.md"
        test_file.write_text("original content")

        old_mtime = test_file.stat().st_mtime - 100
        tracker.track_file(test_file, old_mtime)

        assert tracker.is_file_updated(test_file) is True
