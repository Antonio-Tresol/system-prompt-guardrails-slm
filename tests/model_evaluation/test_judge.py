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
    _extract_json,
    backfill_groundedness,
    backfill_results,
    classify_groundedness,
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
