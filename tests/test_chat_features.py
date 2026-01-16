"""Tests for chat features including QuestionBank and model selection."""

import tempfile
from pathlib import Path

import pytest

from model_evaluation.chat import (
    BYTES_PER_PARAM,
    MODEL_OPTIONS,
    MODEL_PARAMS,
    ModelOption,
    Question,
    QuestionBank,
    QuestionBrowserState,
    VRAMBreakdown,
    create_model_options,
    estimate_vram_gb,
    format_vram_estimate,
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


class TestModelOption:
    """Tests for the ModelOption dataclass."""

    def test_model_option_creation(self) -> None:
        """ModelOption should be created with all required fields."""
        breakdown = VRAMBreakdown(
            model_params_gb=2.0,
            kv_cache_gb=0.5,
            activations_gb=0.3,
            sae_gb=0.5,
            overhead_gb=0.5,
            total_gb=3.8,
        )
        option = ModelOption(
            key="1",
            name="Test Model",
            size="4b",
            quantization="int4",
            description="A test model",
            vram_estimate="~5 GB",
            vram_breakdown=breakdown,
        )
        assert option.key == "1"
        assert option.name == "Test Model"
        assert option.size == "4b"
        assert option.quantization == "int4"
        assert option.description == "A test model"
        assert option.vram_estimate == "~5 GB"
        assert option.vram_breakdown == breakdown

    def test_model_option_with_no_quantization(self) -> None:
        """ModelOption should accept None for quantization (bf16)."""
        breakdown = VRAMBreakdown(
            model_params_gb=7.5,
            kv_cache_gb=0.5,
            activations_gb=0.3,
            sae_gb=0.5,
            overhead_gb=1.3,
            total_gb=10.1,
        )
        option = ModelOption(
            key="2",
            name="Full Precision Model",
            size="4b",
            quantization=None,
            description="No quantization",
            vram_estimate="~10 GB",
            vram_breakdown=breakdown,
        )
        assert option.quantization is None
        assert option.vram_breakdown.model_params_gb == 7.5


class TestVramEstimation:
    """Tests for VRAM estimation functions."""

    def test_estimate_vram_returns_positive_value(self) -> None:
        """estimate_vram_gb should return a positive value."""
        vram = estimate_vram_gb(model_size="4b", quantization=None)
        assert vram > 0

    def test_estimate_vram_bf16_larger_than_int4(self) -> None:
        """bf16 should require more VRAM than int4 for same model size."""
        vram_bf16 = estimate_vram_gb(model_size="4b", quantization=None)
        vram_int4 = estimate_vram_gb(model_size="4b", quantization="int4")
        assert vram_bf16 > vram_int4

    def test_estimate_vram_larger_model_needs_more_vram(self) -> None:
        """Larger models should require more VRAM at same precision."""
        vram_4b = estimate_vram_gb(model_size="4b", quantization="int4")
        vram_12b = estimate_vram_gb(model_size="12b", quantization="int4")
        assert vram_12b > vram_4b

    def test_estimate_vram_4b_bf16_in_expected_range(self) -> None:
        """4B bf16 VRAM should be in reasonable range (8-12 GB)."""
        vram = estimate_vram_gb(model_size="4b", quantization=None)
        assert 8.0 <= vram <= 12.0

    def test_estimate_vram_4b_int4_in_expected_range(self) -> None:
        """4B int4 VRAM should be in reasonable range (2-5 GB)."""
        vram = estimate_vram_gb(model_size="4b", quantization="int4")
        assert 2.0 <= vram <= 5.0

    def test_estimate_vram_12b_int4_in_expected_range(self) -> None:
        """12B int4 VRAM should be in reasonable range (6-12 GB)."""
        vram = estimate_vram_gb(model_size="12b", quantization="int4")
        assert 6.0 <= vram <= 12.0

    def test_format_vram_estimate_includes_gb(self) -> None:
        """format_vram_estimate should include 'GB' in output."""
        result = format_vram_estimate(5.5)
        assert "GB" in result

    def test_format_vram_estimate_includes_tilde(self) -> None:
        """format_vram_estimate should include '~' prefix."""
        result = format_vram_estimate(5.5)
        assert result.startswith("~")

    def test_format_vram_estimate_one_decimal(self) -> None:
        """format_vram_estimate should show one decimal place."""
        result = format_vram_estimate(5.55)
        assert result == "~5.6 GB" or result == "~5.5 GB"  # Rounding

    def test_model_params_has_expected_sizes(self) -> None:
        """MODEL_PARAMS should have standard Gemma sizes."""
        assert "1b" in MODEL_PARAMS
        assert "4b" in MODEL_PARAMS
        assert "12b" in MODEL_PARAMS
        assert "27b" in MODEL_PARAMS

    def test_bytes_per_param_has_expected_precisions(self) -> None:
        """BYTES_PER_PARAM should have expected precision types."""
        assert None in BYTES_PER_PARAM  # bf16 default
        assert "int4" in BYTES_PER_PARAM
        assert "int8" in BYTES_PER_PARAM

    def test_bytes_per_param_values_correct(self) -> None:
        """BYTES_PER_PARAM values should match precision specifications."""
        assert BYTES_PER_PARAM[None] == 2.0  # bf16
        assert BYTES_PER_PARAM["int4"] == 0.5
        assert BYTES_PER_PARAM["int8"] == 1.0


class TestModelOptions:
    """Tests for the predefined MODEL_OPTIONS list."""

    def test_model_options_has_three_entries(self) -> None:
        """MODEL_OPTIONS should have exactly 3 predefined options."""
        assert len(MODEL_OPTIONS) == 3

    def test_model_options_keys_are_unique(self) -> None:
        """Each model option should have a unique key."""
        keys = [opt.key for opt in MODEL_OPTIONS]
        assert len(keys) == len(set(keys))

    def test_model_options_keys_are_sequential(self) -> None:
        """Model option keys should be 1, 2, 3."""
        keys = [opt.key for opt in MODEL_OPTIONS]
        assert keys == ["1", "2", "3"]

    def test_gemma_4b_bf16_option(self) -> None:
        """First option should be Gemma 3 4B IT bf16."""
        option = MODEL_OPTIONS[0]
        assert option.key == "1"
        assert option.size == "4b"
        assert option.quantization is None
        assert "Gemma 3" in option.name
        assert "4B" in option.name
        assert "bf16" in option.name.lower()

    def test_gemma_4b_int4_option(self) -> None:
        """Second option should be Gemma 3 4B IT int4."""
        option = MODEL_OPTIONS[1]
        assert option.key == "2"
        assert option.size == "4b"
        assert option.quantization == "int4"
        assert "Gemma 3" in option.name
        assert "4B" in option.name
        assert "int4" in option.name.lower()

    def test_gemma_12b_int4_option(self) -> None:
        """Third option should be Gemma 3 12B IT int4."""
        option = MODEL_OPTIONS[2]
        assert option.key == "3"
        assert option.size == "12b"
        assert option.quantization == "int4"
        assert "Gemma 3" in option.name
        assert "12B" in option.name
        assert "int4" in option.name.lower()

    def test_all_options_have_vram_estimates(self) -> None:
        """All model options should have VRAM estimates with GB."""
        for option in MODEL_OPTIONS:
            assert option.vram_estimate
            assert "GB" in option.vram_estimate

    def test_all_options_have_vram_breakdown(self) -> None:
        """All model options should have detailed VRAM breakdown."""
        for option in MODEL_OPTIONS:
            breakdown = option.vram_breakdown
            assert breakdown is not None
            assert breakdown.model_params_gb > 0
            assert breakdown.kv_cache_gb >= 0
            assert breakdown.activations_gb >= 0
            assert breakdown.sae_gb > 0
            assert breakdown.overhead_gb > 0
            assert breakdown.total_gb > 0

    def test_vram_breakdown_components_sum_correctly(self) -> None:
        """VRAM breakdown components should sum to total (with overhead)."""
        for option in MODEL_OPTIONS:
            b = option.vram_breakdown
            base = b.model_params_gb + b.kv_cache_gb + b.activations_gb + b.sae_gb
            expected_total = base + b.overhead_gb
            assert abs(b.total_gb - expected_total) < 0.01

    def test_quantized_model_uses_less_vram_than_bf16(self) -> None:
        """4-bit quantized model should use less VRAM than bf16."""
        bf16_option = MODEL_OPTIONS[0]  # Gemma 3 4B IT (bf16)
        int4_option = MODEL_OPTIONS[1]  # Gemma 3 4B IT (int4)
        assert (
            int4_option.vram_breakdown.model_params_gb < bf16_option.vram_breakdown.model_params_gb
        )
        assert int4_option.vram_breakdown.total_gb < bf16_option.vram_breakdown.total_gb

    def test_all_options_have_descriptions(self) -> None:
        """All model options should have descriptions."""
        for option in MODEL_OPTIONS:
            assert option.description
            assert len(option.description) > 0

    def test_create_model_options_returns_list(self) -> None:
        """create_model_options should return a list of ModelOption."""
        options = create_model_options()
        assert isinstance(options, list)
        assert len(options) == 3
        assert all(isinstance(opt, ModelOption) for opt in options)

    def test_vram_estimates_are_computed(self) -> None:
        """VRAM estimates should match computed values."""
        options = create_model_options()
        for opt in options:
            expected_vram = estimate_vram_gb(
                model_size=opt.size,
                quantization=opt.quantization,
            )
            expected_str = format_vram_estimate(expected_vram)
            assert opt.vram_estimate == expected_str
