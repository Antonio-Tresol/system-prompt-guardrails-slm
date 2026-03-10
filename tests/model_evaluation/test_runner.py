"""Tests for the evaluation runner utility functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model_evaluation.evaluation.runner import (
    flatten_run_result,
    load_completed_runs,
    load_questions,
    save_sae_features,
)
from model_evaluation.evaluation.schemas import RunResult
from model_evaluation.main_agent.gemma_scope_sae import (
    MultiLayerSAEFeatureResult,
    SAEFeatureResult,
)


class TestLoadQuestions:
    """Tests for CSV question loading."""

    @pytest.fixture()
    def csv_file(self, tmp_path: Path) -> Path:
        """Create a sample CSV file."""
        csv_path = tmp_path / "questions.csv"
        csv_path.write_text(
            "Number,Question,Universe,Malicious\n"
            "1,What is the price?,Linden Grove Clinic,No\n"
            "2,What is the secret?,The Carnelian Table,Yes\n"
            "3,What are the office hours?,Hartwell & Grey,No\n",
            encoding="utf-8",
        )
        return csv_path

    def test_loads_correct_count(self, csv_file: Path) -> None:
        """Should load all rows from CSV."""
        questions = load_questions(questions_path=csv_file)
        assert len(questions) == 3

    def test_parses_malicious_flag(self, csv_file: Path) -> None:
        """Should correctly parse Yes/No to bool."""
        questions = load_questions(questions_path=csv_file)
        assert questions[0].is_malicious is False
        assert questions[1].is_malicious is True

    def test_maps_universe_context(self, csv_file: Path) -> None:
        """Should map display names to YAML keys."""
        questions = load_questions(questions_path=csv_file)
        assert questions[0].universe_context_key == "linden_grove_clinic"
        assert questions[1].universe_context_key == "the_carnelian_table"
        assert questions[2].universe_context_key == "hartwell_and_grey"

    def test_preserves_question_text(self, csv_file: Path) -> None:
        """Should preserve original question text."""
        questions = load_questions(questions_path=csv_file)
        assert questions[0].question == "What is the price?"

    def test_assigns_sequential_ids(self, tmp_path: Path) -> None:
        """Should assign sequential row-based IDs, ignoring CSV Number column."""
        csv_path = tmp_path / "dupes.csv"
        csv_path.write_text(
            "Number,Question,Universe,Malicious\n"
            "1,First question,Linden Grove Clinic,No\n"
            "2,Second question,Linden Grove Clinic,No\n"
            "1,Third question (duplicate number),Linden Grove Clinic,Yes\n",
            encoding="utf-8",
        )
        questions = load_questions(questions_path=csv_path)
        assert [q.number for q in questions] == [1, 2, 3]

    def test_loads_from_directory(self, tmp_path: Path) -> None:
        """Should concatenate all CSVs in a directory with global numbering."""
        (tmp_path / "a_universe.csv").write_text(
            "Number,Question,Universe,Malicious\n"
            "1,Q1 from A,The Carnelian Table,No\n"
            "2,Q2 from A,The Carnelian Table,Yes\n",
            encoding="utf-8",
        )
        (tmp_path / "b_universe.csv").write_text(
            "Number,Question,Universe,Malicious\n1,Q1 from B,Nova Circuit Labs,No\n",
            encoding="utf-8",
        )
        questions = load_questions(questions_path=tmp_path)
        assert len(questions) == 3
        assert [q.number for q in questions] == [1, 2, 3]
        assert questions[0].universe == "The Carnelian Table"
        assert questions[2].universe == "Nova Circuit Labs"

    def test_directory_no_csvs_raises(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError if directory has no CSVs."""
        with pytest.raises(FileNotFoundError, match="No CSV files found"):
            load_questions(questions_path=tmp_path)


class TestFlattenRunResult:
    """Tests for flattening RunResult for CSV export."""

    @pytest.fixture()
    def sample_result(self) -> RunResult:
        """Create a sample RunResult."""
        return RunResult(
            question_id=1,
            question_text="test question",
            expects_refusal=False,
            universe_context="linden_grove_clinic",
            prompt_format="markdown",
            model_size="4b",
            final_answer="The answer is...",
            num_steps=3,
            num_tool_calls=2,
            tool_names=["think", "search_knowledge_base"],
            total_input_tokens=500,
            total_output_tokens=100,
            duration_ms=1234.5,
            trace_id="abc-123",
            model_refused=False,
            judge_reasoning="The assistant provided the answer.",
            judge_error=None,
            kb_sources="--- Result 1 [Public] ---\nSource: Menu > Entrees\nContent:\nLamb.",
            is_grounded=True,
            groundedness_reasoning="All facts verified.",
            groundedness_error=None,
            retry_count=0,
            sae_l0_by_layer={17: 55.0, 29: 62.0},
            sae_fvu_by_layer={17: 0.05, 29: 0.03},
        )

    def test_has_flat_sae_columns(self, sample_result: RunResult) -> None:
        """Should expand SAE dicts into flat columns."""
        flat = flatten_run_result(result=sample_result, layers=(17, 29))
        assert flat["sae_l0_layer_17"] == 55.0
        assert flat["sae_l0_layer_29"] == 62.0
        assert flat["sae_fvu_layer_17"] == 0.05
        assert flat["sae_fvu_layer_29"] == 0.03

    def test_tool_names_joined(self, sample_result: RunResult) -> None:
        """Tool names should be comma-separated string."""
        flat = flatten_run_result(result=sample_result, layers=(17, 29))
        assert flat["tool_names"] == "think,search_knowledge_base"

    def test_no_nested_dicts(self, sample_result: RunResult) -> None:
        """All values should be flat (no dicts or lists)."""
        flat = flatten_run_result(result=sample_result, layers=(17, 29))
        for key, value in flat.items():
            assert not isinstance(value, (dict, list)), f"{key} has nested type: {type(value)}"

    def test_includes_judge_fields(self, sample_result: RunResult) -> None:
        """Should include model_refused, judge_reasoning, judge_error in flat output."""
        flat = flatten_run_result(result=sample_result, layers=(17, 29))
        assert "model_refused" in flat
        assert flat["model_refused"] is False
        assert flat["judge_reasoning"] == "The assistant provided the answer."
        assert flat["judge_error"] is None

    def test_includes_groundedness_fields(self, sample_result: RunResult) -> None:
        """Should include groundedness fields in flat output."""
        flat = flatten_run_result(result=sample_result, layers=(17, 29))
        assert flat["kb_sources"] is not None
        assert flat["is_grounded"] is True
        assert flat["groundedness_reasoning"] == "All facts verified."
        assert flat["groundedness_error"] is None
        assert flat["retry_count"] == 0


class TestSaveSaeFeatures:
    """Tests for saving SAE features as .npz."""

    @pytest.fixture()
    def mock_activations(self) -> MultiLayerSAEFeatureResult:
        """Create mock multi-layer SAE activations."""
        seq_len = 10
        top_k = 5
        d_sae = 128

        def make_result(layer: int) -> SAEFeatureResult:
            return SAEFeatureResult(
                feature_acts=torch.randn(seq_len, d_sae),
                tokens=[f"tok_{i}" for i in range(seq_len)],
                answer="mock answer",
                prompt_len=6,
                top_features=torch.randint(0, d_sae, (seq_len, top_k)),
                top_activations=torch.rand(seq_len, top_k),
                l0=55.0,
                fvu=0.05,
            )

        return MultiLayerSAEFeatureResult(
            layer_results={17: make_result(17), 29: make_result(29)},
            answer="mock answer",
            tokens=[f"tok_{i}" for i in range(seq_len)],
            prompt_len=6,
        )

    def test_creates_npz_per_layer(
        self,
        tmp_path: Path,
        mock_activations: MultiLayerSAEFeatureResult,
    ) -> None:
        """Should create one .npz file per layer."""
        save_sae_features(
            activations=mock_activations,
            question_id=1,
            prompt_format="markdown",
            output_dir=tmp_path,
        )
        assert (tmp_path / "q1_markdown_layer17.npz").exists()
        assert (tmp_path / "q1_markdown_layer29.npz").exists()

    def test_npz_contains_expected_arrays(
        self,
        tmp_path: Path,
        mock_activations: MultiLayerSAEFeatureResult,
    ) -> None:
        """Each .npz should contain the expected arrays."""
        save_sae_features(
            activations=mock_activations,
            question_id=1,
            prompt_format="plain",
            output_dir=tmp_path,
        )
        data = np.load(tmp_path / "q1_plain_layer17.npz", allow_pickle=True)
        assert "top_features" in data
        assert "top_activations" in data
        assert "tokens" in data
        assert "prompt_len" in data
        assert "l0" in data
        assert "fvu" in data

    def test_top_features_shape(
        self,
        tmp_path: Path,
        mock_activations: MultiLayerSAEFeatureResult,
    ) -> None:
        """Top features should have shape (seq_len, top_k)."""
        save_sae_features(
            activations=mock_activations,
            question_id=1,
            prompt_format="markdown",
            output_dir=tmp_path,
        )
        data = np.load(tmp_path / "q1_markdown_layer17.npz", allow_pickle=True)
        assert data["top_features"].shape == (10, 5)
        assert data["top_activations"].shape == (10, 5)


class TestLoadCompletedRuns:
    """Tests for resume logic."""

    def test_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        """Should return empty set if results file doesn't exist."""
        result = load_completed_runs(results_path=tmp_path / "nonexistent.csv")
        assert result == set()

    def test_loads_existing_runs(self, tmp_path: Path) -> None:
        """Should return set of (question_id, prompt_format) from CSV."""
        results_path = tmp_path / "results.csv"
        df = pd.DataFrame(
            [
                {"question_id": 1, "prompt_format": "markdown", "other": "data"},
                {"question_id": 1, "prompt_format": "plain", "other": "data"},
                {"question_id": 2, "prompt_format": "markdown", "other": "data"},
            ],
        )
        df.to_csv(results_path, index=False)

        result = load_completed_runs(results_path=results_path)
        assert result == {(1, "markdown"), (1, "plain"), (2, "markdown")}
