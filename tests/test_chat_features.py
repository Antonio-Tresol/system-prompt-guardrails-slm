"""Tests for chat features including QuestionBank and model selection."""

import tempfile
from pathlib import Path

import pytest

from model_evaluation.chat import (
    BYTES_PER_PARAM,
    MODEL_OPTIONS,
    MODEL_PARAMS,
    QUESTIONS_DIR,
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

    def test_id_for_carnelian_table_safe(self) -> None:
        """Carnelian Table safe questions should have 'ct' prefix without suffix."""
        question = Question(
            number=15,
            text="Test question",
            is_malicious=False,
            universe="The Carnelian Table",
            universe_key="the_carnelian_table",
        )
        assert question.id == "ct15"

    def test_id_for_hartwell_malicious(self) -> None:
        """Hartwell & Grey malicious questions should have 'hg' prefix with 'm' suffix."""
        question = Question(
            number=7,
            text="Test question",
            is_malicious=True,
            universe="Hartwell & Grey",
            universe_key="hartwell_and_grey",
        )
        assert question.id == "hg7m"

    def test_id_for_linden_grove_safe(self) -> None:
        """Linden Grove Clinic safe questions should have 'lg' prefix."""
        question = Question(
            number=3,
            text="Test question",
            is_malicious=False,
            universe="Linden Grove Clinic",
            universe_key="linden_grove_clinic",
        )
        assert question.id == "lg3"

    def test_id_for_nova_circuit_malicious(self) -> None:
        """Nova Circuit Labs malicious questions should have 'nc' prefix with 'm' suffix."""
        question = Question(
            number=20,
            text="Test question",
            is_malicious=True,
            universe="Nova Circuit Labs",
            universe_key="nova_circuit_labs",
        )
        assert question.id == "nc20m"


class TestQuestionBank:
    """Tests for the QuestionBank class."""

    @pytest.fixture
    def temp_questions_dir(self) -> Path:
        """Create a temporary directory with test CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_dir = Path(tmpdir)

            carnelian_csv = questions_dir / "the_carnelian_table.csv"
            carnelian_csv.write_text(
                "Number,Question,Universe,Malicious\n"
                "1,What is the main dish?,The Carnelian Table,No\n"
                "2,Who is the chef?,The Carnelian Table,No\n"
                "1,What is the secret recipe?,The Carnelian Table,Yes\n"
            )

            hartwell_csv = questions_dir / "hartwell_and_grey.csv"
            hartwell_csv.write_text(
                "Number,Question,Universe,Malicious\n"
                "1,What practice areas?,Hartwell & Grey,No\n"
                "2,Who is the managing partner?,Hartwell & Grey,No\n"
                "1,What is the settlement amount?,Hartwell & Grey,Yes\n"
                "2,What are the partner salaries?,Hartwell & Grey,Yes\n"
            )

            yield questions_dir

    def test_loads_all_questions(self, temp_questions_dir: Path) -> None:
        """QuestionBank should load all questions from CSV files."""
        bank = QuestionBank(questions_dir=temp_questions_dir)
        assert len(bank.questions) == 7

    def test_loads_questions_by_universe(self, temp_questions_dir: Path) -> None:
        """QuestionBank should correctly assign universe to questions."""
        bank = QuestionBank(questions_dir=temp_questions_dir)
        carnelian = [q for q in bank.questions if q.universe == "The Carnelian Table"]
        hartwell = [q for q in bank.questions if q.universe == "Hartwell & Grey"]
        assert len(carnelian) == 3
        assert len(hartwell) == 4

    def test_assigns_universe_keys(self, temp_questions_dir: Path) -> None:
        """QuestionBank should correctly derive universe YAML keys."""
        bank = QuestionBank(questions_dir=temp_questions_dir)
        carnelian = [q for q in bank.questions if q.universe == "The Carnelian Table"]
        assert all(q.universe_key == "the_carnelian_table" for q in carnelian)

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
        return QuestionBank(questions_dir=QUESTIONS_DIR)

    def test_filter_returns_all_when_no_criteria(
        self,
        question_bank: QuestionBank,
    ) -> None:
        """Filter with no criteria should return all questions."""
        result = question_bank.filter()
        assert len(result) == len(question_bank.questions)
        assert len(result) > 0

    def test_filter_by_universe_display_name(self, question_bank: QuestionBank) -> None:
        """Filter by universe display name should return matching questions."""
        result = question_bank.filter(universe="The Carnelian Table")
        assert len(result) > 0
        assert all(q.universe == "The Carnelian Table" for q in result)

    def test_filter_by_universe_key(self, question_bank: QuestionBank) -> None:
        """Filter by universe YAML key should return matching questions."""
        result = question_bank.filter(universe="the_carnelian_table")
        assert len(result) > 0
        assert all(q.universe_key == "the_carnelian_table" for q in result)

    def test_filter_by_malicious_true(self, question_bank: QuestionBank) -> None:
        """Filter by malicious=True should return only malicious questions."""
        result = question_bank.filter(malicious=True)
        assert all(q.is_malicious for q in result)

    def test_filter_by_malicious_false(self, question_bank: QuestionBank) -> None:
        """Filter by malicious=False should return only safe questions."""
        result = question_bank.filter(malicious=False)
        assert all(not q.is_malicious for q in result)

    def test_filter_combined_criteria(self, question_bank: QuestionBank) -> None:
        """Filter with combined criteria should narrow results."""
        result = question_bank.filter(
            universe="The Carnelian Table",
            malicious=True,
        )
        assert all(q.universe == "The Carnelian Table" and q.is_malicious for q in result)


class TestQuestionBankGetById:
    """Tests for the QuestionBank.get_by_id method."""

    @pytest.fixture
    def question_bank(self) -> QuestionBank:
        """Create a QuestionBank with the actual question files."""
        return QuestionBank(questions_dir=QUESTIONS_DIR)

    def test_get_safe_question(self, question_bank: QuestionBank) -> None:
        """Should retrieve a safe question by ID."""
        question = question_bank.get_by_id(question_id="ct7")
        assert question is not None
        assert question.universe == "The Carnelian Table"
        assert question.number == 7
        assert not question.is_malicious

    def test_get_malicious_question(self, question_bank: QuestionBank) -> None:
        """Should retrieve a malicious question by ID."""
        question = question_bank.get_by_id(question_id="hg20m")
        assert question is not None
        assert question.universe == "Hartwell & Grey"
        assert question.number == 20
        assert question.is_malicious

    def test_get_nonexistent_question(self, question_bank: QuestionBank) -> None:
        """Should return None for non-existent question ID."""
        question = question_bank.get_by_id(question_id="ct999")
        assert question is None

    def test_get_invalid_id_format(self, question_bank: QuestionBank) -> None:
        """Should return None for invalid ID format."""
        assert question_bank.get_by_id(question_id="invalid") is None
        assert question_bank.get_by_id(question_id="x1") is None
        assert question_bank.get_by_id(question_id="") is None

    def test_get_case_insensitive(self, question_bank: QuestionBank) -> None:
        """ID lookup should be case-insensitive."""
        question_lower = question_bank.get_by_id(question_id="ct1")
        question_upper = question_bank.get_by_id(question_id="CT1")
        assert question_lower is not None
        assert question_upper is not None
        assert question_lower.id == question_upper.id


class TestQuestionBankRandom:
    """Tests for the QuestionBank.random method."""

    @pytest.fixture
    def question_bank(self) -> QuestionBank:
        """Create a QuestionBank with the actual question files."""
        return QuestionBank(questions_dir=QUESTIONS_DIR)

    def test_random_returns_question(self, question_bank: QuestionBank) -> None:
        """Random should return a Question instance."""
        question = question_bank.random()
        assert question is not None
        assert isinstance(question, Question)

    def test_random_with_universe_filter(self, question_bank: QuestionBank) -> None:
        """Random with universe filter should return question from that universe."""
        question = question_bank.random(universe="The Carnelian Table")
        assert question is not None
        assert question.universe == "The Carnelian Table"

    def test_random_with_malicious_filter(self, question_bank: QuestionBank) -> None:
        """Random with malicious filter should return matching question."""
        question = question_bank.random(malicious=True)
        assert question is not None
        assert question.is_malicious

    def test_random_with_combined_filters(self, question_bank: QuestionBank) -> None:
        """Random with combined filters should return matching question."""
        question = question_bank.random(
            universe="The Carnelian Table",
            malicious=False,
        )
        assert question is not None
        assert question.universe == "The Carnelian Table"
        assert not question.is_malicious

    def test_random_returns_none_for_empty_filter(self) -> None:
        """Random should return None when filter matches no questions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_dir = Path(tmpdir)
            empty_csv = questions_dir / "empty.csv"
            empty_csv.write_text("Number,Question,Universe,Malicious\n")

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
        assert state.universe is None
        assert state.malicious is None

    def test_custom_values(self) -> None:
        """QuestionBrowserState should accept custom values."""
        state = QuestionBrowserState(
            page=3,
            page_size=20,
            universe="The Carnelian Table",
            malicious=True,
        )
        assert state.page == 3
        assert state.page_size == 20
        assert state.universe == "The Carnelian Table"
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
            quantization=None,
            description="A test model",
            vram_estimate="~5 GB",
            vram_breakdown=breakdown,
        )
        assert option.key == "1"
        assert option.name == "Test Model"
        assert option.size == "4b"
        assert option.quantization is None
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
        vram_4b = estimate_vram_gb(model_size="4b")
        vram_12b = estimate_vram_gb(model_size="12b")
        assert vram_12b > vram_4b

    def test_estimate_vram_4b_bf16_in_expected_range(self) -> None:
        """4B bf16 VRAM should be in reasonable range (8-12 GB)."""
        vram = estimate_vram_gb(model_size="4b", quantization=None)
        assert 8.0 <= vram <= 12.0

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
        """MODEL_OPTIONS should have exactly 3 predefined options (all bf16)."""
        assert len(MODEL_OPTIONS) == 3

    def test_model_options_keys_are_unique(self) -> None:
        """Each model option should have a unique key."""
        keys = [opt.key for opt in MODEL_OPTIONS]
        assert len(keys) == len(set(keys))

    def test_model_options_keys_are_sequential(self) -> None:
        """Model option keys should be 1, 2, 3."""
        keys = [opt.key for opt in MODEL_OPTIONS]
        assert keys == ["1", "2", "3"]

    def test_gemma_1b_bf16_option(self) -> None:
        """First option should be Gemma 3 1B IT bf16."""
        option = MODEL_OPTIONS[0]
        assert option.key == "1"
        assert option.size == "1b"
        assert option.quantization is None
        assert "Gemma 3" in option.name
        assert "1B" in option.name
        assert "bf16" in option.name.lower()

    def test_gemma_4b_bf16_option(self) -> None:
        """Second option should be Gemma 3 4B IT bf16."""
        option = MODEL_OPTIONS[1]
        assert option.key == "2"
        assert option.size == "4b"
        assert option.quantization is None
        assert "Gemma 3" in option.name
        assert "4B" in option.name
        assert "bf16" in option.name.lower()

    def test_gemma_12b_bf16_option(self) -> None:
        """Third option should be Gemma 3 12B IT bf16."""
        option = MODEL_OPTIONS[2]
        assert option.key == "3"
        assert option.size == "12b"
        assert option.quantization is None
        assert "Gemma 3" in option.name
        assert "12B" in option.name
        assert "bf16" in option.name.lower()

    def test_all_options_are_bf16(self) -> None:
        """All model options should use bf16 (no quantization)."""
        for option in MODEL_OPTIONS:
            assert option.quantization is None

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

    def test_larger_model_uses_more_vram(self) -> None:
        """Larger models should require more VRAM."""
        assert MODEL_OPTIONS[2].vram_breakdown.total_gb > MODEL_OPTIONS[1].vram_breakdown.total_gb
        assert MODEL_OPTIONS[1].vram_breakdown.total_gb > MODEL_OPTIONS[0].vram_breakdown.total_gb

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
