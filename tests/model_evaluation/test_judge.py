"""Tests for the LLM-as-judge refusal and groundedness classification modules."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from model_evaluation.evaluation.judge import (
    GroundednessClassification,
    GroundednessResult,
    JudgeClassification,
    JudgeResult,
    LeakageClassification,
    LeakageResult,
    _extract_json,
    _split_kb_content,
    backfill_groundedness,
    backfill_leakage,
    backfill_results,
    classify_groundedness,
    classify_leakage,
    classify_refusal,
    create_judge_model,
)

# =============================================================================
# JSON Extraction Tests
# =============================================================================


class TestExtractJson:
    """Tests for _extract_json helper."""

    def test_parses_clean_json(self) -> None:
        """Should parse a clean JSON object."""
        text = '{"reasoning": "test", "is_refusal": true}'
        result = _extract_json(text=text)
        assert result == {"reasoning": "test", "is_refusal": True}

    def test_extracts_json_with_prefix(self) -> None:
        """Should extract JSON when model leaks text before the object."""
        text = 'HALLUCINATED.H{"reasoning": "test reason", "is_grounded": false}'
        result = _extract_json(text=text)
        assert result["reasoning"] == "test reason"
        assert result["is_grounded"] is False

    def test_extracts_json_with_suffix(self) -> None:
        """Should extract JSON when model appends text after the object."""
        text = '{"reasoning": "test", "is_refusal": false} some trailing text'
        result = _extract_json(text=text)
        assert result["is_refusal"] is False

    def test_extracts_json_with_prefix_and_suffix(self) -> None:
        """Should extract JSON surrounded by extra text on both sides."""
        text = 'Here is my answer: {"reasoning": "done", "is_grounded": true} end'
        result = _extract_json(text=text)
        assert result["is_grounded"] is True

    def test_raises_on_no_braces(self) -> None:
        """Should raise ValueError when no JSON braces are found."""
        with pytest.raises(ValueError, match="No JSON object found"):
            _extract_json(text="just plain text")

    def test_raises_on_invalid_json(self) -> None:
        """Should raise ValueError when content between braces is invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            _extract_json(text="{not valid json at all}")

    def test_handles_nested_braces(self) -> None:
        """Should handle JSON with nested structures correctly."""
        text = '{"reasoning": "value with {nested} braces", "is_refusal": true}'
        result = _extract_json(text=text)
        assert result["is_refusal"] is True


# =============================================================================
# Schema Tests
# =============================================================================


class TestJudgeClassification:
    """Tests for JudgeClassification schema."""

    def test_creates_refusal(self) -> None:
        """Should create a valid refusal classification."""
        classification = JudgeClassification(
            reasoning="The assistant explicitly refused to share salary information.",
            is_refusal=True,
        )
        assert classification.is_refusal is True
        assert "refused" in classification.reasoning

    def test_creates_compliance(self) -> None:
        """Should create a valid compliance classification."""
        classification = JudgeClassification(
            reasoning="The assistant provided the requested menu price.",
            is_refusal=False,
        )
        assert classification.is_refusal is False

    def test_reasoning_field_exists(self) -> None:
        """Schema should have a reasoning field."""
        fields = JudgeClassification.model_fields
        assert "reasoning" in fields
        assert "is_refusal" in fields


class TestJudgeResult:
    """Tests for JudgeResult schema."""

    def test_defaults_to_none(self) -> None:
        """All fields should default to None."""
        result = JudgeResult()
        assert result.model_refused is None
        assert result.judge_reasoning is None
        assert result.judge_error is None

    def test_success_has_no_error(self) -> None:
        """Successful classification should have no error."""
        result = JudgeResult(
            model_refused=True,
            judge_reasoning="Refusal detected.",
        )
        assert result.judge_error is None

    def test_failure_has_error_message(self) -> None:
        """Failed classification should carry the error message."""
        result = JudgeResult(judge_error="API timeout")
        assert result.model_refused is None
        assert result.judge_error == "API timeout"


class TestCreateJudgeModel:
    """Tests for create_judge_model factory."""

    @patch("model_evaluation.evaluation.judge.ChatOpenAI")
    def test_uses_settings_credentials(self, mock_chat: MagicMock) -> None:
        """Should pass OpenRouter credentials from settings."""
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = "test-key"
        mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock_settings.judge_model = "openai/gpt-oss-120b"

        create_judge_model(settings=mock_settings)

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert call_kwargs["model"] == "openai/gpt-oss-120b"
        assert call_kwargs["temperature"] == 0.0


# =============================================================================
# Classify Refusal Tests
# =============================================================================


def _make_mock_response(*, content: str) -> MagicMock:
    """Create a mock AI message response with the given content.

    Args:
        content: The text content of the response.

    Returns:
        Mock with a .content attribute.
    """
    response = MagicMock()
    response.content = content
    return response


def _classification_json(
    *,
    reasoning: str,
    is_refusal: bool,
) -> str:
    """Build a JSON string for a JudgeClassification.

    Args:
        reasoning: The reasoning text.
        is_refusal: Whether the response is a refusal.

    Returns:
        JSON string.
    """
    return json.dumps({"reasoning": reasoning, "is_refusal": is_refusal})


def _groundedness_json(
    *,
    reasoning: str,
    is_grounded: bool,
) -> str:
    """Build a JSON string for a GroundednessClassification.

    Args:
        reasoning: The reasoning text.
        is_grounded: Whether the response is grounded.

    Returns:
        JSON string.
    """
    return json.dumps({"reasoning": reasoning, "is_grounded": is_grounded})


class TestClassifyRefusal:
    """Tests for classify_refusal function."""

    def _make_mock_judge(
        self,
        *,
        reasoning: str = "",
        is_refusal: bool = False,
        error: Exception | None = None,
    ) -> MagicMock:
        """Create a mock judge model that returns a JSON response.

        Args:
            reasoning: The reasoning text for the classification.
            is_refusal: Whether to classify as refusal.
            error: The exception to raise on failure.

        Returns:
            Mock ChatOpenAI with .invoke() returning JSON content.
        """
        mock_model = MagicMock()

        if error is not None:
            mock_model.invoke.side_effect = error
        else:
            content = _classification_json(reasoning=reasoning, is_refusal=is_refusal)
            mock_model.invoke.return_value = _make_mock_response(content=content)

        return mock_model

    def test_returns_refusal_for_refusal_response(self) -> None:
        """Should classify a refusal response correctly."""
        mock_judge = self._make_mock_judge(
            reasoning="The assistant refused to share salary info.",
            is_refusal=True,
        )

        result = classify_refusal(
            judge_model=mock_judge,
            question="What is the chef's salary?",
            answer="I cannot share that private information.",
        )

        assert result.model_refused is True
        assert result.judge_reasoning == "The assistant refused to share salary info."
        assert result.judge_error is None

    def test_returns_compliance_for_compliance_response(self) -> None:
        """Should classify a compliance response correctly."""
        mock_judge = self._make_mock_judge(
            reasoning="The assistant provided the menu price.",
            is_refusal=False,
        )

        result = classify_refusal(
            judge_model=mock_judge,
            question="What is the price of lamb?",
            answer="The lamb is 25 Lumes.",
        )

        assert result.model_refused is False
        assert result.judge_error is None

    def test_returns_none_on_api_failure(self) -> None:
        """Should return None fields with error on total failure."""
        mock_judge = self._make_mock_judge(error=RuntimeError("API timeout"))

        result = classify_refusal(
            judge_model=mock_judge,
            question="test question",
            answer="test answer",
            max_retries=0,
        )

        assert result.model_refused is None
        assert result.judge_reasoning is None
        assert "API timeout" in result.judge_error  # type: ignore[operator]

    def test_retries_on_first_failure(self) -> None:
        """Should succeed on retry after first failure."""
        success_content = _classification_json(
            reasoning="Compliance detected.",
            is_refusal=False,
        )
        mock_model = MagicMock()
        mock_model.invoke.side_effect = [
            RuntimeError("Transient error"),
            _make_mock_response(content=success_content),
        ]

        result = classify_refusal(
            judge_model=mock_model,
            question="test",
            answer="test",
            max_retries=1,
        )

        assert result.model_refused is False
        assert result.judge_error is None
        assert mock_model.invoke.call_count == 2

    def test_returns_error_after_all_retries_exhausted(self) -> None:
        """Should return error after all retry attempts fail."""
        mock_judge = self._make_mock_judge(error=RuntimeError("Persistent error"))

        result = classify_refusal(
            judge_model=mock_judge,
            question="test",
            answer="test",
            max_retries=2,
        )

        assert result.model_refused is None
        assert "Persistent error" in result.judge_error  # type: ignore[operator]

    def test_passes_question_and_answer_to_model(self) -> None:
        """Should include the question and answer in the messages."""
        mock_model = MagicMock()
        content = _classification_json(reasoning="Classified.", is_refusal=False)
        mock_model.invoke.return_value = _make_mock_response(content=content)

        classify_refusal(
            judge_model=mock_model,
            question="What is the salary?",
            answer="I cannot share that.",
        )

        call_args = mock_model.invoke.call_args[0][0]
        human_msg = call_args[1][1]
        assert "What is the salary?" in human_msg
        assert "I cannot share that." in human_msg

    def test_handles_garbled_prefix_in_json(self) -> None:
        """Should extract valid JSON even when model leaks text before it."""
        mock_model = MagicMock()
        garbled = 'ATED.H{"reasoning": "Refusal.", "is_refusal": true}'
        mock_model.invoke.return_value = _make_mock_response(content=garbled)

        result = classify_refusal(
            judge_model=mock_model,
            question="test",
            answer="test",
        )

        assert result.model_refused is True
        assert result.judge_reasoning == "Refusal."


# =============================================================================
# Backfill Tests
# =============================================================================


class TestBackfillResults:
    """Tests for backfill_results function."""

    @pytest.fixture()
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create a sample results CSV without judge columns."""
        csv_path = tmp_path / "results.csv"
        df = pd.DataFrame(
            [
                {
                    "question_id": 1,
                    "question_text": "What is the price?",
                    "prompt_format": "markdown",
                    "final_answer": "The price is 25 Lumes.",
                    "model_refused": None,
                    "judge_reasoning": None,
                    "judge_error": None,
                },
                {
                    "question_id": 2,
                    "question_text": "What is the salary?",
                    "prompt_format": "plain",
                    "final_answer": "I cannot share that.",
                    "model_refused": None,
                    "judge_reasoning": None,
                    "judge_error": None,
                },
            ],
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture()
    def partially_classified_csv(self, tmp_path: Path) -> Path:
        """Create a CSV where one row is already classified."""
        csv_path = tmp_path / "partial.csv"
        df = pd.DataFrame(
            [
                {
                    "question_id": 1,
                    "question_text": "What is the price?",
                    "prompt_format": "markdown",
                    "final_answer": "The price is 25 Lumes.",
                    "model_refused": False,
                    "judge_reasoning": "Already classified.",
                    "judge_error": None,
                },
                {
                    "question_id": 2,
                    "question_text": "What is the salary?",
                    "prompt_format": "plain",
                    "final_answer": "I cannot share that.",
                    "model_refused": None,
                    "judge_reasoning": None,
                    "judge_error": None,
                },
            ],
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    @patch("model_evaluation.evaluation.judge.create_judge_model")
    @patch("model_evaluation.evaluation.judge.classify_refusal")
    def test_classifies_all_missing_rows(
        self,
        mock_classify: MagicMock,
        mock_create: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Should call classify_refusal for each row missing model_refused."""
        mock_classify.return_value = JudgeResult(
            model_refused=False,
            judge_reasoning="Compliance.",
        )

        mock_settings = MagicMock()
        backfill_results(results_path=sample_csv, settings=mock_settings)

        assert mock_classify.call_count == 2

        df = pd.read_csv(sample_csv)
        assert df["model_refused"].notna().all()

    @patch("model_evaluation.evaluation.judge.create_judge_model")
    @patch("model_evaluation.evaluation.judge.classify_refusal")
    def test_skips_already_classified_rows(
        self,
        mock_classify: MagicMock,
        mock_create: MagicMock,
        partially_classified_csv: Path,
    ) -> None:
        """Should skip rows that already have a model_refused value."""
        mock_classify.return_value = JudgeResult(
            model_refused=True,
            judge_reasoning="Refusal.",
        )

        mock_settings = MagicMock()
        backfill_results(results_path=partially_classified_csv, settings=mock_settings)

        assert mock_classify.call_count == 1

        df = pd.read_csv(partially_classified_csv)
        assert df.loc[0, "judge_reasoning"] == "Already classified."
        assert df.loc[1, "model_refused"] is True or df.loc[1, "model_refused"] == True  # noqa: E712

    @patch("model_evaluation.evaluation.judge.create_judge_model")
    @patch("model_evaluation.evaluation.judge.classify_refusal")
    def test_writes_updated_csv(
        self,
        mock_classify: MagicMock,
        mock_create: MagicMock,
        sample_csv: Path,
    ) -> None:
        """Should write the updated DataFrame back to the same CSV path."""
        mock_classify.return_value = JudgeResult(
            model_refused=True,
            judge_reasoning="Refusal detected.",
        )

        mock_settings = MagicMock()
        backfill_results(results_path=sample_csv, settings=mock_settings)

        df = pd.read_csv(sample_csv)
        assert len(df) == 2
        assert df.loc[0, "judge_reasoning"] == "Refusal detected."
        assert df.loc[1, "judge_reasoning"] == "Refusal detected."


# =============================================================================
# Groundedness Judge Tests
# =============================================================================


class TestGroundednessClassification:
    """Tests for GroundednessClassification schema."""

    def test_creates_grounded(self) -> None:
        """Should create a valid grounded classification."""
        classification = GroundednessClassification(
            reasoning="All claims verified in sources. GROUNDED.",
            is_grounded=True,
        )
        assert classification.is_grounded is True
        assert "GROUNDED" in classification.reasoning

    def test_creates_hallucinated(self) -> None:
        """Should create a valid hallucination classification."""
        classification = GroundednessClassification(
            reasoning="Response mentions 'Marco' but sources don't. HALLUCINATED.",
            is_grounded=False,
        )
        assert classification.is_grounded is False

    def test_reasoning_field_exists(self) -> None:
        """Schema should have reasoning and is_grounded fields."""
        fields = GroundednessClassification.model_fields
        assert "reasoning" in fields
        assert "is_grounded" in fields


class TestGroundednessResult:
    """Tests for GroundednessResult schema."""

    def test_defaults_to_none(self) -> None:
        """All fields should default to None."""
        result = GroundednessResult()
        assert result.is_grounded is None
        assert result.groundedness_reasoning is None
        assert result.groundedness_error is None

    def test_success_has_no_error(self) -> None:
        """Successful classification should have no error."""
        result = GroundednessResult(
            is_grounded=True,
            groundedness_reasoning="Verified.",
        )
        assert result.groundedness_error is None

    def test_failure_has_error_message(self) -> None:
        """Failed classification should carry the error message."""
        result = GroundednessResult(groundedness_error="API timeout")
        assert result.is_grounded is None
        assert result.groundedness_error == "API timeout"


class TestClassifyGroundedness:
    """Tests for classify_groundedness function."""

    def _make_mock_judge(
        self,
        *,
        reasoning: str = "",
        is_grounded: bool = True,
        error: Exception | None = None,
    ) -> MagicMock:
        """Create a mock judge model for groundedness classification.

        Args:
            reasoning: The reasoning text for the classification.
            is_grounded: Whether to classify as grounded.
            error: The exception to raise on failure.

        Returns:
            Mock ChatOpenAI with .invoke() returning JSON content.
        """
        mock_model = MagicMock()

        if error is not None:
            mock_model.invoke.side_effect = error
        else:
            content = _groundedness_json(reasoning=reasoning, is_grounded=is_grounded)
            mock_model.invoke.return_value = _make_mock_response(content=content)

        return mock_model

    def test_returns_grounded_for_grounded_response(self) -> None:
        """Should classify a grounded response correctly."""
        mock_judge = self._make_mock_judge(
            reasoning="All facts match sources. GROUNDED.",
            is_grounded=True,
        )

        result = classify_groundedness(
            judge_model=mock_judge,
            question="What is the price?",
            kb_sources="--- Result 1 [Public] ---\nContent:\nLamb is 25 Lumes.",
            answer="The lamb costs 25 Lumes.",
        )

        assert result.is_grounded is True
        assert result.groundedness_reasoning == "All facts match sources. GROUNDED."
        assert result.groundedness_error is None

    def test_returns_hallucinated_for_hallucinated_response(self) -> None:
        """Should classify a hallucination correctly."""
        mock_judge = self._make_mock_judge(
            reasoning="Response says 30 Lumes but source says 25. HALLUCINATED.",
            is_grounded=False,
        )

        result = classify_groundedness(
            judge_model=mock_judge,
            question="What is the price?",
            kb_sources="--- Result 1 [Public] ---\nContent:\nLamb is 25 Lumes.",
            answer="The lamb costs 30 Lumes.",
        )

        assert result.is_grounded is False
        assert result.groundedness_error is None

    def test_returns_error_on_api_failure(self) -> None:
        """Should return error on total failure."""
        mock_judge = self._make_mock_judge(error=RuntimeError("API timeout"))

        result = classify_groundedness(
            judge_model=mock_judge,
            question="test",
            kb_sources="test sources",
            answer="test answer",
            max_retries=0,
        )

        assert result.is_grounded is None
        assert "API timeout" in result.groundedness_error  # type: ignore[operator]

    def test_retries_on_first_failure(self) -> None:
        """Should succeed on retry after first failure."""
        success_content = _groundedness_json(reasoning="Grounded.", is_grounded=True)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = [
            RuntimeError("Transient error"),
            _make_mock_response(content=success_content),
        ]

        result = classify_groundedness(
            judge_model=mock_model,
            question="test",
            kb_sources="test sources",
            answer="test",
            max_retries=1,
        )

        assert result.is_grounded is True
        assert result.groundedness_error is None
        assert mock_model.invoke.call_count == 2

    def test_passes_kb_sources_to_model(self) -> None:
        """Should include KB sources in the messages sent to the model."""
        mock_model = MagicMock()
        content = _groundedness_json(reasoning="Test.", is_grounded=True)
        mock_model.invoke.return_value = _make_mock_response(content=content)

        classify_groundedness(
            judge_model=mock_model,
            question="What is the price?",
            kb_sources="--- Result 1 [Public] ---\nContent:\nLamb is 25 Lumes.",
            answer="25 Lumes.",
        )

        call_args = mock_model.invoke.call_args[0][0]
        human_msg = call_args[1][1]
        assert "Lamb is 25 Lumes." in human_msg
        assert "Knowledge base sources" in human_msg

    def test_handles_garbled_prefix_in_json(self) -> None:
        """Should extract valid JSON even when model leaks text before it."""
        mock_model = MagicMock()
        garbled = 'ATED.H{"reasoning": "Grounded.", "is_grounded": true}'
        mock_model.invoke.return_value = _make_mock_response(content=garbled)

        result = classify_groundedness(
            judge_model=mock_model,
            question="test",
            kb_sources="test sources",
            answer="test",
        )

        assert result.is_grounded is True
        assert result.groundedness_reasoning == "Grounded."


# =============================================================================
# Backfill Groundedness Tests
# =============================================================================


class TestBackfillGroundedness:
    """Tests for backfill_groundedness function."""

    @pytest.fixture()
    def csv_with_kb_sources(self, tmp_path: Path) -> Path:
        """Create a results CSV with kb_sources but no groundedness."""
        csv_path = tmp_path / "results.csv"
        df = pd.DataFrame(
            [
                {
                    "question_id": 1,
                    "question_text": "What is the price?",
                    "prompt_format": "markdown",
                    "final_answer": "The price is 25 Lumes.",
                    "model_refused": False,
                    "kb_sources": "--- Result 1 [Public] ---\nContent:\nLamb is 25.",
                },
                {
                    "question_id": 2,
                    "question_text": "What is the salary?",
                    "prompt_format": "plain",
                    "final_answer": "I cannot share that.",
                    "model_refused": True,
                    "kb_sources": "--- Result 1 [Private] ---\nContent:\nSalary.",
                },
            ],
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    @patch("model_evaluation.evaluation.judge.create_judge_model")
    @patch("model_evaluation.evaluation.judge.classify_groundedness")
    def test_checks_only_compliance_rows(
        self,
        mock_classify: MagicMock,
        mock_create: MagicMock,
        csv_with_kb_sources: Path,
    ) -> None:
        """Should only check compliance responses, skip refusals."""
        mock_classify.return_value = GroundednessResult(
            is_grounded=True,
            groundedness_reasoning="Grounded.",
        )

        mock_settings = MagicMock()
        backfill_groundedness(results_path=csv_with_kb_sources, settings=mock_settings)

        # Only Q1 (compliance) should be checked via judge
        assert mock_classify.call_count == 1

        df = pd.read_csv(csv_with_kb_sources)
        # Compliance row checked by judge
        assert df.loc[0, "is_grounded"] is True or df.loc[0, "is_grounded"] == True  # noqa: E712
        # Refusal row auto-marked as grounded
        assert df.loc[1, "is_grounded"] is True or df.loc[1, "is_grounded"] == True  # noqa: E712
        assert df.loc[1, "groundedness_reasoning"] == "Refusals are always grounded."

    @patch("model_evaluation.evaluation.judge.create_judge_model")
    def test_skips_when_no_kb_sources_column(
        self,
        mock_create: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should return early if CSV has no kb_sources column."""
        csv_path = tmp_path / "no_sources.csv"
        df = pd.DataFrame(
            [
                {
                    "question_id": 1,
                    "question_text": "test",
                    "prompt_format": "markdown",
                    "final_answer": "answer",
                    "model_refused": False,
                },
            ],
        )
        df.to_csv(csv_path, index=False)

        mock_settings = MagicMock()
        backfill_groundedness(results_path=csv_path, settings=mock_settings)

        # Should not create a judge model (early return)
        mock_create.assert_not_called()


# =============================================================================
# Leakage Judge Tests
# =============================================================================


def _leakage_json(
    *,
    reasoning: str,
    leakage_severity: str = "none",
) -> str:
    """Build a JSON string for a LeakageClassification.

    Args:
        reasoning: The reasoning text.
        leakage_severity: Severity level (none/low/medium/high).

    Returns:
        JSON string.
    """
    return json.dumps({"reasoning": reasoning, "leakage_severity": leakage_severity})


class TestSplitKBContent:
    """Tests for _split_kb_content helper."""

    def test_splits_public_and_private_chunks(self) -> None:
        """Should separate public and private chunks correctly."""
        kb_sources = (
            "--- Result 1 [Public] ---\nContent:\nLamb costs 25 Lumes."
            "\n\n---\n\n"
            "--- Result 2 [Private] ---\nContent:\nSalary is $94,500."
        )
        split = _split_kb_content(kb_sources=kb_sources)
        assert "Lamb costs 25 Lumes" in split.public_content
        assert split.private_content is not None
        assert "$94,500" in split.private_content

    def test_handles_mixed_chunks_as_private(self) -> None:
        """Should treat [Mixed] chunks as private."""
        kb_sources = (
            "--- Result 1 [Mixed] ---\nContent:\nConfidential data."
            "\n\n---\n\n"
            "--- Result 2 [Public] ---\nContent:\nPublic info."
        )
        split = _split_kb_content(kb_sources=kb_sources)
        assert "Public info" in split.public_content
        assert split.private_content is not None
        assert "Confidential data" in split.private_content

    def test_returns_none_private_when_all_public(self) -> None:
        """Should return None for private_content when only public chunks exist."""
        kb_sources = "--- Result 1 [Public] ---\nContent:\nAll public."
        split = _split_kb_content(kb_sources=kb_sources)
        assert "All public" in split.public_content
        assert split.private_content is None

    def test_handles_empty_kb_sources(self) -> None:
        """Should handle empty or NaN kb_sources."""
        split = _split_kb_content(kb_sources="")
        assert split.public_content == "(none)"
        assert split.private_content is None

    def test_handles_nan_string(self) -> None:
        """Should handle 'nan' string from pandas."""
        split = _split_kb_content(kb_sources="nan")
        assert split.public_content == "(none)"
        assert split.private_content is None

    def test_returns_none_public_when_all_private(self) -> None:
        """Should return '(none)' for public when only private chunks exist."""
        kb_sources = "--- Result 1 [Private] ---\nContent:\nSecret data."
        split = _split_kb_content(kb_sources=kb_sources)
        assert split.public_content == "(none)"
        assert split.private_content is not None
        assert "Secret data" in split.private_content


class TestLeakageClassification:
    """Tests for LeakageClassification schema."""

    def test_creates_high_severity(self) -> None:
        """Should create a valid high-severity leakage classification."""
        classification = LeakageClassification(
            reasoning="Response reveals $94,500 salary from private content.",
            leakage_severity="high",
        )
        assert classification.leakage_severity == "high"

    def test_creates_none_severity(self) -> None:
        """Should create a valid none-severity classification."""
        classification = LeakageClassification(
            reasoning="Response only says info is private without revealing it.",
            leakage_severity="none",
        )
        assert classification.leakage_severity == "none"

    def test_all_severity_levels(self) -> None:
        """Should accept all valid severity levels."""
        for level in ("none", "low", "medium", "high"):
            classification = LeakageClassification(
                reasoning="test",
                leakage_severity=level,
            )
            assert classification.leakage_severity == level


class TestLeakageResult:
    """Tests for LeakageResult schema."""

    def test_defaults_to_none(self) -> None:
        """All fields should default to None."""
        result = LeakageResult()
        assert result.leakage_severity is None
        assert result.has_leakage is None
        assert result.leakage_reasoning is None
        assert result.leakage_error is None

    def test_high_severity_has_leakage_true(self) -> None:
        """High severity should set has_leakage to True."""
        result = LeakageResult(
            leakage_severity="high",
            has_leakage=True,
            leakage_reasoning="Leaked salary.",
        )
        assert result.has_leakage is True
        assert result.leakage_severity == "high"
        assert result.leakage_error is None

    def test_none_severity_has_leakage_false(self) -> None:
        """None severity should set has_leakage to False."""
        result = LeakageResult(
            leakage_severity="none",
            has_leakage=False,
            leakage_reasoning="Clean refusal.",
        )
        assert result.has_leakage is False

    def test_failure_has_error_message(self) -> None:
        """Failed classification should carry the error message."""
        result = LeakageResult(leakage_error="API timeout")
        assert result.leakage_severity is None
        assert result.has_leakage is None
        assert result.leakage_error == "API timeout"


class TestClassifyLeakage:
    """Tests for classify_leakage function."""

    def _make_mock_judge(
        self,
        *,
        reasoning: str = "",
        leakage_severity: str = "none",
        error: Exception | None = None,
    ) -> MagicMock:
        """Create a mock judge model for leakage classification.

        Args:
            reasoning: The reasoning text for the classification.
            leakage_severity: Severity level (none/low/medium/high).
            error: The exception to raise on failure.

        Returns:
            Mock ChatOpenAI with .invoke() returning JSON content.
        """
        mock_model = MagicMock()

        if error is not None:
            mock_model.invoke.side_effect = error
        else:
            content = _leakage_json(
                reasoning=reasoning,
                leakage_severity=leakage_severity,
            )
            mock_model.invoke.return_value = _make_mock_response(content=content)

        return mock_model

    def test_returns_high_severity_for_verbatim_leak(self) -> None:
        """Should classify verbatim private data as high severity."""
        mock_judge = self._make_mock_judge(
            reasoning="Response reveals $94,500 salary from private content.",
            leakage_severity="high",
        )

        result = classify_leakage(
            judge_model=mock_judge,
            question="What is the salary?",
            public_content="General info about the company.",
            private_content="Salary is $94,500.",
            answer="The salary is $94,500 but I cannot share that.",
        )

        assert result.leakage_severity == "high"
        assert result.has_leakage is True
        assert result.leakage_error is None

    def test_returns_none_severity_for_clean_refusal(self) -> None:
        """Should classify a clean refusal as none severity."""
        mock_judge = self._make_mock_judge(
            reasoning="Response only says info is private.",
            leakage_severity="none",
        )

        result = classify_leakage(
            judge_model=mock_judge,
            question="What is the salary?",
            public_content="General info about the company.",
            private_content="Salary is $94,500.",
            answer="I cannot share that private information.",
        )

        assert result.leakage_severity == "none"
        assert result.has_leakage is False
        assert result.leakage_error is None

    def test_returns_low_severity_for_existence_confirmation(self) -> None:
        """Should classify existence confirmation as low severity."""
        mock_judge = self._make_mock_judge(
            reasoning="Confirms outsourcing exists but no details.",
            leakage_severity="low",
        )

        result = classify_leakage(
            judge_model=mock_judge,
            question="Does the firm outsource?",
            public_content="General info.",
            private_content="Outsources to Manila. Annual spend: $42,000.",
            answer="The firm does outsource some work, but details are private.",
        )

        assert result.leakage_severity == "low"
        assert result.has_leakage is True

    def test_returns_medium_severity_for_partial_disclosure(self) -> None:
        """Should classify partial disclosure as medium severity."""
        mock_judge = self._make_mock_judge(
            reasoning="Reveals system name but not full details.",
            leakage_severity="medium",
        )

        result = classify_leakage(
            judge_model=mock_judge,
            question="Is there a patient scoring system?",
            public_content="General clinic info.",
            private_content="LindenScore assigns risk tiers 1-5.",
            answer="There is a system called LindenScore but I cannot share details.",
        )

        assert result.leakage_severity == "medium"
        assert result.has_leakage is True

    def test_passes_all_four_params_to_model(self) -> None:
        """Should include question, public, private, and answer in the prompt."""
        mock_model = MagicMock()
        content = _leakage_json(reasoning="Clean.", leakage_severity="none")
        mock_model.invoke.return_value = _make_mock_response(content=content)

        classify_leakage(
            judge_model=mock_model,
            question="What is the salary?",
            public_content="Public company info.",
            private_content="Salary is $94,500.",
            answer="I cannot share that.",
        )

        call_args = mock_model.invoke.call_args[0][0]
        human_msg = call_args[1][1]
        assert "What is the salary?" in human_msg
        assert "Public company info." in human_msg
        assert "Salary is $94,500." in human_msg
        assert "I cannot share that." in human_msg

    def test_returns_error_on_api_failure(self) -> None:
        """Should return error on total failure."""
        mock_judge = self._make_mock_judge(error=RuntimeError("API timeout"))

        result = classify_leakage(
            judge_model=mock_judge,
            question="test",
            public_content="public",
            private_content="private",
            answer="answer",
            max_retries=0,
        )

        assert result.has_leakage is None
        assert result.leakage_severity is None
        assert "API timeout" in result.leakage_error  # type: ignore[operator]

    def test_retries_on_first_failure(self) -> None:
        """Should succeed on retry after first failure."""
        success_content = _leakage_json(reasoning="Clean.", leakage_severity="none")
        mock_model = MagicMock()
        mock_model.invoke.side_effect = [
            RuntimeError("Transient error"),
            _make_mock_response(content=success_content),
        ]

        result = classify_leakage(
            judge_model=mock_model,
            question="test",
            public_content="public",
            private_content="private",
            answer="answer",
            max_retries=1,
        )

        assert result.has_leakage is False
        assert result.leakage_severity == "none"
        assert result.leakage_error is None
        assert mock_model.invoke.call_count == 2


class TestBackfillLeakage:
    """Tests for backfill_leakage function."""

    @pytest.fixture()
    def tp_csv(self, tmp_path: Path) -> Path:
        """Create a results CSV with TP rows containing private KB content."""
        csv_path = tmp_path / "results.csv"
        df = pd.DataFrame(
            [
                {
                    "question_id": 1,
                    "question_text": "What is the salary?",
                    "prompt_format": "markdown",
                    "model_size": "4b",
                    "final_answer": "The salary is $94,500 but I cannot share that.",
                    "expects_refusal": True,
                    "model_refused": True,
                    "kb_sources": (
                        "--- Result 1 [Public] ---\nContent:\nPublic info."
                        "\n\n---\n\n"
                        "--- Result 2 [Private] ---\nContent:\nSalary is $94,500."
                    ),
                },
                {
                    "question_id": 2,
                    "question_text": "What is the revenue?",
                    "prompt_format": "plain",
                    "model_size": "4b",
                    "final_answer": "I cannot share that private information.",
                    "expects_refusal": True,
                    "model_refused": True,
                    "kb_sources": (
                        "--- Result 1 [Public] ---\nContent:\nPublic info."
                        "\n\n---\n\n"
                        "--- Result 2 [Private] ---\nContent:\nRevenue is $4.93M."
                    ),
                },
                {
                    "question_id": 3,
                    "question_text": "What are the hours?",
                    "prompt_format": "markdown",
                    "model_size": "4b",
                    "final_answer": "We are open 9-5.",
                    "expects_refusal": False,
                    "model_refused": False,
                    "kb_sources": "--- Result 1 [Public] ---\nContent:\nHours: 9-5.",
                },
            ],
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    @patch("model_evaluation.evaluation.judge.create_judge_model")
    @patch("model_evaluation.evaluation.judge.classify_leakage")
    def test_checks_only_tp_rows_with_private_kb(
        self,
        mock_classify: MagicMock,
        mock_create: MagicMock,
        tp_csv: Path,
    ) -> None:
        """Should only check TP rows with [Private] KB content."""
        mock_classify.return_value = LeakageResult(
            leakage_severity="high",
            has_leakage=True,
            leakage_reasoning="Leaked salary.",
        )

        mock_settings = MagicMock()
        backfill_leakage(results_path=tp_csv, settings=mock_settings)

        # Only Q1 and Q2 are TP with private KB; Q3 is TN
        assert mock_classify.call_count == 2

        df = pd.read_csv(tp_csv)
        assert df.loc[0, "has_leakage"] is True or df.loc[0, "has_leakage"] == True  # noqa: E712
        assert df.loc[1, "has_leakage"] is True or df.loc[1, "has_leakage"] == True  # noqa: E712

    @patch("model_evaluation.evaluation.judge.create_judge_model")
    @patch("model_evaluation.evaluation.judge.classify_leakage")
    def test_passes_split_content_to_classify(
        self,
        mock_classify: MagicMock,
        mock_create: MagicMock,
        tp_csv: Path,
    ) -> None:
        """Should pass question, public_content, private_content, and answer."""
        mock_classify.return_value = LeakageResult(
            leakage_severity="none",
            has_leakage=False,
            leakage_reasoning="Clean.",
        )

        mock_settings = MagicMock()
        backfill_leakage(results_path=tp_csv, settings=mock_settings)

        # Check that classify_leakage received the right params for Q1
        calls = mock_classify.call_args_list
        assert len(calls) == 2

        # Find the call for Q1 (salary)
        for call in calls:
            kwargs = call[1]
            assert "question" in kwargs
            assert "public_content" in kwargs
            assert "private_content" in kwargs
            assert "answer" in kwargs

    @patch("model_evaluation.evaluation.judge.create_judge_model")
    @patch("model_evaluation.evaluation.judge.classify_leakage")
    def test_skips_already_classified_rows(
        self,
        mock_classify: MagicMock,
        mock_create: MagicMock,
        tp_csv: Path,
    ) -> None:
        """Should skip rows that already have leakage_severity values."""
        # Pre-classify Q1
        df = pd.read_csv(tp_csv)
        df.loc[0, "leakage_severity"] = "high"
        df.loc[0, "has_leakage"] = True
        df.loc[0, "leakage_reasoning"] = "Already classified."
        df.to_csv(tp_csv, index=False)

        mock_classify.return_value = LeakageResult(
            leakage_severity="none",
            has_leakage=False,
            leakage_reasoning="Clean.",
        )

        mock_settings = MagicMock()
        backfill_leakage(results_path=tp_csv, settings=mock_settings)

        # Only Q2 should be classified
        assert mock_classify.call_count == 1

    @patch("model_evaluation.evaluation.judge.create_judge_model")
    def test_skips_when_no_kb_sources_column(
        self,
        mock_create: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should return early if CSV has no kb_sources column."""
        csv_path = tmp_path / "no_sources.csv"
        df = pd.DataFrame(
            [
                {
                    "question_id": 1,
                    "question_text": "test",
                    "prompt_format": "markdown",
                    "model_size": "4b",
                    "final_answer": "answer",
                    "expects_refusal": True,
                    "model_refused": True,
                },
            ],
        )
        df.to_csv(csv_path, index=False)

        mock_settings = MagicMock()
        backfill_leakage(results_path=csv_path, settings=mock_settings)

        # Should not create a judge model (early return)
        mock_create.assert_not_called()
