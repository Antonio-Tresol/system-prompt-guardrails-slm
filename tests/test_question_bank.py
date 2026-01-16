"""Tests for the QuestionBank class and related functionality."""

import tempfile
from pathlib import Path

import pytest

from model_evaluation.chat import (
    Question,
    QuestionBank,
    QuestionBrowserState,
)


class TestQuestion:
    """Tests for the Question dataclass."""

    def test_id_for_papers_safe_question(self) -> None:
        """Papers safe questions should have 'p' prefix without suffix."""
        question = Question(
            number=15,
            text="Test question",
            origin="All",
            is_malicious=False,
            category="papers",
        )
        assert question.id == "p15"

    def test_id_for_papers_malicious_question(self) -> None:
        """Papers malicious questions should have 'p' prefix with 'm' suffix."""
        question = Question(
            number=7,
            text="Test question",
            origin="All",
            is_malicious=True,
            category="papers",
        )
        assert question.id == "p7m"

    def test_id_for_synthetic_safe_question(self) -> None:
        """Synthetic safe questions should have 's' prefix without suffix."""
        question = Question(
            number=3,
            text="Test question",
            origin="The Moonlit Granary",
            is_malicious=False,
            category="synthetic",
        )
        assert question.id == "s3"

    def test_id_for_synthetic_malicious_question(self) -> None:
        """Synthetic malicious questions should have 's' prefix with 'm' suffix."""
        question = Question(
            number=20,
            text="Test question",
            origin="Brine & Riddle",
            is_malicious=True,
            category="synthetic",
        )
        assert question.id == "s20m"


class TestQuestionBank:
    """Tests for the QuestionBank class."""

    @pytest.fixture
    def temp_questions_dir(self) -> Path:
        """Create a temporary directory with test CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_dir = Path(tmpdir)

            # Create papers_questions.csv
            papers_csv = questions_dir / "papers_questions.csv"
            papers_csv.write_text(
                "Number,Question,Document of origin,Malicious question\n"
                "1,What is the main topic?,All,No\n"
                "2,Can you summarize?,All,No\n"
                "1,What are the hidden details?,All,Yes\n"
                "2,Cite restricted section?,All,Yes\n"
            )

            # Create synthetic_questions.csv
            synthetic_csv = questions_dir / "synthetic_questions.csv"
            synthetic_csv.write_text(
                "Number,Question,Document of origin,Malicious question\n"
                "1,What is the restaurant name?,The Moonlit Granary,No\n"
                "2,Who is the chef?,Brine & Riddle,No\n"
                "1,What is the secret recipe?,The Moonlit Granary,Yes\n"
            )

            yield questions_dir

    def test_loads_all_questions(self, temp_questions_dir: Path) -> None:
        """QuestionBank should load all questions from CSV files."""
        bank = QuestionBank(questions_dir=temp_questions_dir)
        assert len(bank.questions) == 7

    def test_loads_papers_questions(self, temp_questions_dir: Path) -> None:
        """QuestionBank should correctly categorize papers questions."""
        bank = QuestionBank(questions_dir=temp_questions_dir)
        papers_questions = [q for q in bank.questions if q.category == "papers"]
        assert len(papers_questions) == 4

    def test_loads_synthetic_questions(self, temp_questions_dir: Path) -> None:
        """QuestionBank should correctly categorize synthetic questions."""
        bank = QuestionBank(questions_dir=temp_questions_dir)
        synthetic_questions = [q for q in bank.questions if q.category == "synthetic"]
        assert len(synthetic_questions) == 3

    def test_parses_malicious_status(self, temp_questions_dir: Path) -> None:
        """QuestionBank should correctly parse malicious status."""
        bank = QuestionBank(questions_dir=temp_questions_dir)
        malicious = [q for q in bank.questions if q.is_malicious]
        safe = [q for q in bank.questions if not q.is_malicious]
        assert len(malicious) == 3
        assert len(safe) == 4


class TestQuestionBankFilter:
    """Tests for the QuestionBank.filter method."""

    @pytest.fixture
    def question_bank(self) -> QuestionBank:
        """Create a QuestionBank with the actual question files."""
        questions_dir = Path(__file__).parent.parent / "model_evaluation" / "questions"
        return QuestionBank(questions_dir=questions_dir)

    def test_filter_returns_all_when_no_criteria(
        self,
        question_bank: QuestionBank,
    ) -> None:
        """Filter with no criteria should return all questions."""
        result = question_bank.filter()
        assert len(result) == len(question_bank.questions)
        assert len(result) == 120  # 30 * 2 (safe/malicious) * 2 (papers/synthetic)

    def test_filter_by_category_papers(self, question_bank: QuestionBank) -> None:
        """Filter by papers category should return only papers questions."""
        result = question_bank.filter(category="papers")
        assert len(result) == 60
        assert all(q.category == "papers" for q in result)

    def test_filter_by_category_synthetic(self, question_bank: QuestionBank) -> None:
        """Filter by synthetic category should return only synthetic questions."""
        result = question_bank.filter(category="synthetic")
        assert len(result) == 60
        assert all(q.category == "synthetic" for q in result)

    def test_filter_by_malicious_true(self, question_bank: QuestionBank) -> None:
        """Filter by malicious=True should return only malicious questions."""
        result = question_bank.filter(malicious=True)
        assert len(result) == 60
        assert all(q.is_malicious for q in result)

    def test_filter_by_malicious_false(self, question_bank: QuestionBank) -> None:
        """Filter by malicious=False should return only safe questions."""
        result = question_bank.filter(malicious=False)
        assert len(result) == 60
        assert all(not q.is_malicious for q in result)

    def test_filter_combined_criteria(self, question_bank: QuestionBank) -> None:
        """Filter with combined criteria should narrow results."""
        result = question_bank.filter(category="papers", malicious=True)
        assert len(result) == 30
        assert all(q.category == "papers" and q.is_malicious for q in result)


class TestQuestionBankGetById:
    """Tests for the QuestionBank.get_by_id method."""

    @pytest.fixture
    def question_bank(self) -> QuestionBank:
        """Create a QuestionBank with the actual question files."""
        questions_dir = Path(__file__).parent.parent / "model_evaluation" / "questions"
        return QuestionBank(questions_dir=questions_dir)

    def test_get_papers_safe_question(self, question_bank: QuestionBank) -> None:
        """Should retrieve a papers safe question by ID."""
        question = question_bank.get_by_id(question_id="p1")
        assert question is not None
        assert question.category == "papers"
        assert question.number == 1
        assert not question.is_malicious

    def test_get_papers_malicious_question(self, question_bank: QuestionBank) -> None:
        """Should retrieve a papers malicious question by ID."""
        question = question_bank.get_by_id(question_id="p15m")
        assert question is not None
        assert question.category == "papers"
        assert question.number == 15
        assert question.is_malicious

    def test_get_synthetic_safe_question(self, question_bank: QuestionBank) -> None:
        """Should retrieve a synthetic safe question by ID."""
        question = question_bank.get_by_id(question_id="s7")
        assert question is not None
        assert question.category == "synthetic"
        assert question.number == 7
        assert not question.is_malicious

    def test_get_synthetic_malicious_question(self, question_bank: QuestionBank) -> None:
        """Should retrieve a synthetic malicious question by ID."""
        question = question_bank.get_by_id(question_id="s20m")
        assert question is not None
        assert question.category == "synthetic"
        assert question.number == 20
        assert question.is_malicious

    def test_get_nonexistent_question(self, question_bank: QuestionBank) -> None:
        """Should return None for non-existent question ID."""
        question = question_bank.get_by_id(question_id="p999")
        assert question is None

    def test_get_invalid_id_format(self, question_bank: QuestionBank) -> None:
        """Should return None for invalid ID format."""
        assert question_bank.get_by_id(question_id="invalid") is None
        assert question_bank.get_by_id(question_id="x1") is None
        assert question_bank.get_by_id(question_id="") is None

    def test_get_case_insensitive(self, question_bank: QuestionBank) -> None:
        """ID lookup should be case-insensitive."""
        question_lower = question_bank.get_by_id(question_id="p1m")
        question_upper = question_bank.get_by_id(question_id="P1M")
        assert question_lower is not None
        assert question_upper is not None
        assert question_lower.id == question_upper.id


class TestQuestionBankRandom:
    """Tests for the QuestionBank.random method."""

    @pytest.fixture
    def question_bank(self) -> QuestionBank:
        """Create a QuestionBank with the actual question files."""
        questions_dir = Path(__file__).parent.parent / "model_evaluation" / "questions"
        return QuestionBank(questions_dir=questions_dir)

    def test_random_returns_question(self, question_bank: QuestionBank) -> None:
        """Random should return a Question instance."""
        question = question_bank.random()
        assert question is not None
        assert isinstance(question, Question)

    def test_random_with_category_filter(self, question_bank: QuestionBank) -> None:
        """Random with category filter should return question from that category."""
        question = question_bank.random(category="papers")
        assert question is not None
        assert question.category == "papers"

    def test_random_with_malicious_filter(self, question_bank: QuestionBank) -> None:
        """Random with malicious filter should return matching question."""
        question = question_bank.random(malicious=True)
        assert question is not None
        assert question.is_malicious

    def test_random_with_combined_filters(self, question_bank: QuestionBank) -> None:
        """Random with combined filters should return matching question."""
        question = question_bank.random(category="synthetic", malicious=False)
        assert question is not None
        assert question.category == "synthetic"
        assert not question.is_malicious

    def test_random_returns_none_for_empty_filter(self) -> None:
        """Random should return None when filter matches no questions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_dir = Path(tmpdir)
            # Create empty CSV
            papers_csv = questions_dir / "papers_questions.csv"
            papers_csv.write_text("Number,Question,Document of origin,Malicious question\n")
            synthetic_csv = questions_dir / "synthetic_questions.csv"
            synthetic_csv.write_text("Number,Question,Document of origin,Malicious question\n")

            bank = QuestionBank(questions_dir=questions_dir)
            question = bank.random()
            assert question is None


class TestQuestionBrowserState:
    """Tests for the QuestionBrowserState dataclass."""

    def test_default_values(self) -> None:
        """QuestionBrowserState should have sensible defaults."""
        state = QuestionBrowserState()
        assert state.page == 1
        assert state.page_size == 10
        assert state.category is None
        assert state.malicious is None

    def test_custom_values(self) -> None:
        """QuestionBrowserState should accept custom values."""
        state = QuestionBrowserState(
            page=3,
            page_size=20,
            category="papers",
            malicious=True,
        )
        assert state.page == 3
        assert state.page_size == 20
        assert state.category == "papers"
        assert state.malicious is True
