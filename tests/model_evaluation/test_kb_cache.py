"""Tests for KB caching module."""

from pathlib import Path

import pytest

from model_evaluation.evaluation.kb_cache import (
    CachedGeneratorSession,
    load_kb_cache,
    save_kb_cache,
)
from model_evaluation.main_agent.kb_generator.schemas import (
    DocumentChunk,
    GeneratorOutput,
)


@pytest.fixture()
def sample_output() -> GeneratorOutput:
    """Create a sample GeneratorOutput for testing."""
    return GeneratorOutput(
        chunks=[
            DocumentChunk(
                document_title="Test Restaurant Menu",
                section="Appetizers",
                subsection="Spring Collection",
                privacy_level="public",
                content="A delicious spring salad with fresh greens.",
            ),
            DocumentChunk(
                document_title="Test Restaurant Menu",
                section="Staff",
                privacy_level="private",
                content="Chef salary is 50000 Lumes annually.",
            ),
        ],
    )


class TestCachedGeneratorSession:
    """Tests for CachedGeneratorSession."""

    def test_returns_cached_output(self, sample_output: GeneratorOutput) -> None:
        """Should return the same output regardless of query."""
        session = CachedGeneratorSession(cached_output=sample_output)

        result = session.generate(
            query="anything",
            include_private_info=True,
            universe_context="moonlit_granary",
        )

        assert result == sample_output

    def test_ignores_query_parameter(self, sample_output: GeneratorOutput) -> None:
        """Different queries should return the same cached output."""
        session = CachedGeneratorSession(cached_output=sample_output)

        result_a = session.generate(
            query="first query",
            include_private_info=True,
        )
        result_b = session.generate(
            query="completely different query",
            include_private_info=False,
            universe_context="carnelian_table",
        )

        assert result_a == result_b

    def test_reset_is_noop(self, sample_output: GeneratorOutput) -> None:
        """Reset should not affect the cached output."""
        session = CachedGeneratorSession(cached_output=sample_output)
        session.reset()

        result = session.generate(
            query="test",
            include_private_info=True,
        )
        assert result == sample_output


class TestKBCacheSerialization:
    """Tests for save/load KB cache round-trip."""

    def test_save_and_load_roundtrip(
        self,
        tmp_path: Path,
        sample_output: GeneratorOutput,
    ) -> None:
        """Saving and loading should produce identical content."""
        cache = {1: sample_output, 2: sample_output}
        cache_path = tmp_path / "kb_cache.json"

        save_kb_cache(cache=cache, output_path=cache_path)
        loaded = load_kb_cache(cache_path=cache_path)

        assert set(loaded.keys()) == {1, 2}
        assert loaded[1] == sample_output
        assert loaded[2] == sample_output

    def test_preserves_privacy_levels(
        self,
        tmp_path: Path,
        sample_output: GeneratorOutput,
    ) -> None:
        """Privacy levels should survive serialization."""
        cache_path = tmp_path / "kb_cache.json"
        save_kb_cache(cache={1: sample_output}, output_path=cache_path)
        loaded = load_kb_cache(cache_path=cache_path)

        assert loaded[1].chunks[0].privacy_level == "public"
        assert loaded[1].chunks[1].privacy_level == "private"

    def test_cache_file_is_valid_json(
        self,
        tmp_path: Path,
        sample_output: GeneratorOutput,
    ) -> None:
        """The saved file should be readable JSON."""
        import json

        cache_path = tmp_path / "kb_cache.json"
        save_kb_cache(cache={1: sample_output}, output_path=cache_path)

        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "1" in data
        assert "chunks" in data["1"]

    def test_creates_parent_directories(
        self,
        tmp_path: Path,
        sample_output: GeneratorOutput,
    ) -> None:
        """Should create parent dirs if they don't exist."""
        cache_path = tmp_path / "nested" / "dir" / "cache.json"
        save_kb_cache(cache={1: sample_output}, output_path=cache_path)
        assert cache_path.exists()
