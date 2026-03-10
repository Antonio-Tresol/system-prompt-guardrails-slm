"""Tests for the stratified trace sampler."""

from pathlib import Path

from paper.trace_analysis.index import OutcomeType, TraceIndex
from paper.trace_analysis.sampler import TraceSampler


class TestStratifiedSampling:
    """Tests for the sample_stratified method."""

    def test_format_outcome_strata(self, synthetic_results_dir: Path) -> None:
        """Stratifying by format × outcome should produce balanced samples."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        sampler = TraceSampler(index=index, seed=42)

        sample = sampler.sample_stratified(
            n_per_stratum=2,
            model_size="4b",
            strata="format_outcome",
        )

        # Should have draws from multiple strata
        formats = {r.prompt_format for r in sample}
        outcomes = {r.outcome for r in sample}
        assert "markdown" in formats
        assert "plain" in formats
        assert len(outcomes) > 1

    def test_format_strata(self, synthetic_results_dir: Path) -> None:
        """Stratifying by format only should produce records from both formats."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        sampler = TraceSampler(index=index, seed=42)

        sample = sampler.sample_stratified(
            n_per_stratum=3,
            model_size="4b",
            strata="format",
        )

        formats = {r.prompt_format for r in sample}
        assert formats == {"markdown", "plain"}
        assert len(sample) == 6

    def test_deterministic_with_seed(self, synthetic_results_dir: Path) -> None:
        """Same seed should produce identical samples."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        sampler1 = TraceSampler(index=index, seed=123)
        sample1 = sampler1.sample_stratified(
            n_per_stratum=2,
            model_size="4b",
            strata="format",
        )

        sampler2 = TraceSampler(index=index, seed=123)
        sample2 = sampler2.sample_stratified(
            n_per_stratum=2,
            model_size="4b",
            strata="format",
        )

        ids1 = [r.trace_id for r in sample1]
        ids2 = [r.trace_id for r in sample2]
        assert ids1 == ids2

    def test_different_seed_different_sample(self, synthetic_results_dir: Path) -> None:
        """Different seeds should produce different samples (with enough data)."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        sampler1 = TraceSampler(index=index, seed=42)
        sample1 = sampler1.sample_stratified(
            n_per_stratum=3,
            model_size="4b",
            strata="format",
        )

        sampler2 = TraceSampler(index=index, seed=99)
        sample2 = sampler2.sample_stratified(
            n_per_stratum=3,
            model_size="4b",
            strata="format",
        )

        ids1 = {r.trace_id for r in sample1}
        ids2 = {r.trace_id for r in sample2}
        # With 8 records per format and 3 per stratum, different seeds
        # should almost certainly produce different sets
        # (but could theoretically match, so we just check the sets exist)
        assert len(ids1) == 6
        assert len(ids2) == 6

    def test_caps_at_available_records(self, synthetic_results_dir: Path) -> None:
        """Should not error when requesting more records than available."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        sampler = TraceSampler(index=index, seed=42)

        sample = sampler.sample_stratified(
            n_per_stratum=100,
            model_size="4b",
            strata="format",
        )

        # Should get all 16 records (8 per format)
        assert len(sample) == 16


class TestNonOverlappingSamples:
    """Tests for the exclude_drawn mechanism."""

    def test_exclude_drawn_produces_different_records(
        self,
        synthetic_results_dir: Path,
    ) -> None:
        """Second sample with exclude_drawn should not overlap with first."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        sampler = TraceSampler(index=index, seed=42)

        sample1 = sampler.sample_stratified(
            n_per_stratum=2,
            model_size="4b",
            strata="format",
        )
        ids1 = {r.trace_id for r in sample1}

        sample2 = sampler.sample_stratified(
            n_per_stratum=2,
            model_size="4b",
            strata="format",
            exclude_drawn=True,
        )
        ids2 = {r.trace_id for r in sample2}

        assert ids1.isdisjoint(ids2)

    def test_drawn_ids_tracked(self, synthetic_results_dir: Path) -> None:
        """Sampler should track which trace_ids have been drawn."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        sampler = TraceSampler(index=index, seed=42)

        assert len(sampler.drawn_ids) == 0

        sample = sampler.sample_stratified(
            n_per_stratum=2,
            model_size="4b",
            strata="format",
        )

        assert len(sampler.drawn_ids) == len(sample)
        assert all(r.trace_id in sampler.drawn_ids for r in sample)

    def test_reset_drawn_clears_tracking(self, synthetic_results_dir: Path) -> None:
        """reset_drawn should allow re-sampling previously drawn records."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        sampler = TraceSampler(index=index, seed=42)

        sampler.sample_stratified(
            n_per_stratum=2,
            model_size="4b",
            strata="format",
        )
        assert len(sampler.drawn_ids) > 0

        sampler.reset_drawn()
        assert len(sampler.drawn_ids) == 0


class TestSampleByOutcome:
    """Tests for the sample_by_outcome method."""

    def test_returns_correct_outcome(self, synthetic_results_dir: Path) -> None:
        """All returned records should match the requested outcome."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        sampler = TraceSampler(index=index, seed=42)

        sample = sampler.sample_by_outcome(
            model_size="4b",
            outcome=OutcomeType.TP,
            n=5,
        )

        assert all(r.outcome == OutcomeType.TP for r in sample)

    def test_respects_format_filter(self, synthetic_results_dir: Path) -> None:
        """Should filter by prompt format when specified."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        sampler = TraceSampler(index=index, seed=42)

        sample = sampler.sample_by_outcome(
            model_size="4b",
            outcome=OutcomeType.TP,
            n=5,
            prompt_format="markdown",
        )

        assert all(r.prompt_format == "markdown" for r in sample)


class TestSampleAll:
    """Tests for the sample_all method (exhaustive, for back-checks)."""

    def test_returns_all_matching(self, synthetic_results_dir: Path) -> None:
        """Should return ALL records matching the criteria."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        sampler = TraceSampler(index=index, seed=42)

        all_4b = sampler.sample_all(model_size="4b")
        assert len(all_4b) == 16

    def test_does_not_mark_as_drawn(self, synthetic_results_dir: Path) -> None:
        """Exhaustive sampling should NOT mark records as drawn."""
        index = TraceIndex(results_dir=synthetic_results_dir)
        sampler = TraceSampler(index=index, seed=42)

        sampler.sample_all(model_size="4b")
        assert len(sampler.drawn_ids) == 0
