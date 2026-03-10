"""Tests for the TraceIndex: loading, filtering, pairing, and discordant detection."""

from pathlib import Path

from paper.trace_analysis.index import OutcomeType, TraceIndex


class TestTraceIndexBuilding:
    """Tests for index construction from synthetic results directory."""

    def test_loads_all_records(self, synthetic_results_dir: Path) -> None:
        """Index should load all trace records from all model directories."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        # 2 models × 8 questions × 2 formats = 32 records
        assert len(index.records) == 32

    def test_discovers_both_model_sizes(self, synthetic_results_dir: Path) -> None:
        """Index should find both model size directories."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        assert index.model_sizes == ["12b", "4b"]

    def test_builds_pairs_per_question(self, synthetic_results_dir: Path) -> None:
        """Each question should have a pair with both MD and Plain traces."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        # 2 models × 8 questions = 16 pairs
        assert len(index.pairs) == 16

        for pair in index.pairs.values():
            assert "markdown" in pair
            assert "plain" in pair

    def test_derives_outcome_correctly(self, synthetic_results_dir: Path) -> None:
        """Outcome types should be correctly derived from expects_refusal × model_refused."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        # Q1: public, both comply → TN
        q1_md = index.pairs[("4b", 1)]["markdown"]
        assert q1_md.outcome == OutcomeType.TN

        # Q2: private, both refuse → TP
        q2_md = index.pairs[("4b", 2)]["markdown"]
        assert q2_md.outcome == OutcomeType.TP

        # Q3: private, MD refuses, Plain doesn't → MD=TP, Plain=FN
        q3_md = index.pairs[("4b", 3)]["markdown"]
        q3_pl = index.pairs[("4b", 3)]["plain"]
        assert q3_md.outcome == OutcomeType.TP
        assert q3_pl.outcome == OutcomeType.FN

        # Q7: public, MD refuses, Plain doesn't → MD=FP, Plain=TN
        q7_md = index.pairs[("4b", 7)]["markdown"]
        q7_pl = index.pairs[("4b", 7)]["plain"]
        assert q7_md.outcome == OutcomeType.FP
        assert q7_pl.outcome == OutcomeType.TN


class TestTraceIndexFiltering:
    """Tests for the filter method."""

    def test_filter_by_model_size(self, synthetic_results_dir: Path) -> None:
        """Filter should return only records for the specified model size."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        results_4b = index.filter(model_size="4b")
        assert len(results_4b) == 16
        assert all(r.model_size == "4b" for r in results_4b)

    def test_filter_by_format(self, synthetic_results_dir: Path) -> None:
        """Filter should return only records for the specified format."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        results_md = index.filter(model_size="4b", prompt_format="markdown")
        assert len(results_md) == 8
        assert all(r.prompt_format == "markdown" for r in results_md)

    def test_filter_by_outcome(self, synthetic_results_dir: Path) -> None:
        """Filter should return only records matching the outcome type."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        tp_records = index.filter(model_size="4b", outcome=OutcomeType.TP)
        assert all(r.outcome == OutcomeType.TP for r in tp_records)
        assert all(r.expects_refusal for r in tp_records)
        assert all(r.model_refused for r in tp_records)

    def test_filter_by_universe(self, synthetic_results_dir: Path) -> None:
        """Filter should return only records for the specified universe."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        results = index.filter(
            model_size="4b",
            universe_context="the_carnelian_table",
        )
        assert all(r.universe_context == "the_carnelian_table" for r in results)

    def test_combined_filters(self, synthetic_results_dir: Path) -> None:
        """Multiple filters should be combined with AND logic."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        results = index.filter(
            model_size="4b",
            prompt_format="markdown",
            outcome=OutcomeType.TP,
        )
        assert all(
            r.model_size == "4b" and r.prompt_format == "markdown" and r.outcome == OutcomeType.TP
            for r in results
        )


class TestDiscordantPairs:
    """Tests for discordant and concordant pair detection."""

    def test_finds_discordant_pairs(self, synthetic_results_dir: Path) -> None:
        """Should identify pairs where MD and Plain have different outcomes."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        discordant = index.get_discordant_pairs(model_size="4b")

        # Q3: MD=TP, Plain=FN; Q7: MD=FP, Plain=TN; Q8: MD=FN, Plain=TP
        assert len(discordant) == 3

        for md_rec, pl_rec in discordant:
            assert md_rec.prompt_format == "markdown"
            assert pl_rec.prompt_format == "plain"
            assert md_rec.outcome != pl_rec.outcome

    def test_discordant_pairs_contain_correct_questions(
        self,
        synthetic_results_dir: Path,
    ) -> None:
        """Discordant pairs should be Q3, Q7, and Q8."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        discordant = index.get_discordant_pairs(model_size="4b")
        discordant_qids = {md.question_id for md, _ in discordant}
        assert discordant_qids == {3, 7, 8}

    def test_finds_concordant_pairs(self, synthetic_results_dir: Path) -> None:
        """Should identify pairs where MD and Plain have the same outcome."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        concordant = index.get_concordant_pairs(model_size="4b")
        assert len(concordant) == 5  # Q1, Q2, Q4, Q5, Q6

    def test_concordant_filtered_by_outcome(self, synthetic_results_dir: Path) -> None:
        """Concordant pairs should be filterable by outcome type."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        tp_concordant = index.get_concordant_pairs(
            model_size="4b",
            outcome=OutcomeType.TP,
        )
        for md_rec, pl_rec in tp_concordant:
            assert md_rec.outcome == OutcomeType.TP
            assert pl_rec.outcome == OutcomeType.TP


class TestTraceIndexSummary:
    """Tests for the summary method."""

    def test_summary_contains_counts(self, synthetic_results_dir: Path) -> None:
        """Summary should contain record counts and discordant pair counts."""
        index = TraceIndex(results_dir=synthetic_results_dir)

        summary = index.summary()
        assert "32 records" in summary
        assert "16 pairs" in summary
        assert "4b" in summary
        assert "12b" in summary
