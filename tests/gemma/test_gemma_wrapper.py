"""Unit tests for GemmaWithSAE wrapper logic.

These tests mock the model, tokenizer, and SAE to verify the wrapper's
message formatting, tool call parsing, and generation logic.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from model_evaluation.main_agent.gemma_scope_sae import SAEConfig, SAEFeatureResult
from model_evaluation.main_agent.gemma_wrapper import GemmaWithSAE

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock HuggingFace model."""
    model = MagicMock()
    # Mock parameters() to return a tensor with device info
    mock_param = torch.zeros(1)
    model.parameters.return_value = iter([mock_param])
    # Mock generate to return tensor output
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    return model


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted prompt"

    # Create a mock that properly handles .to() call
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_inputs.input_ids = torch.tensor([[1, 2, 3]])
    mock_inputs.__getitem__ = lambda self, key: getattr(self, key)
    tokenizer.return_value = mock_inputs

    tokenizer.decode.return_value = "Generated response"
    return tokenizer


@pytest.fixture
def mock_sae() -> MagicMock:
    """Create a mock SAE."""
    sae = MagicMock()
    sae.encode.return_value = torch.zeros(10, 1024)
    return sae


@pytest.fixture
def mock_sae_config() -> SAEConfig:
    """Create a mock SAE config."""
    return SAEConfig(
        model_size="4b",
        model_type="it",
        layer=29,
        width="16k",
        l0_size="medium",
        d_in=2048,
        d_sae=16384,
    )


@pytest.fixture
def wrapper(
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
    mock_sae: MagicMock,
    mock_sae_config: SAEConfig,
) -> GemmaWithSAE:
    """Create a GemmaWithSAE wrapper for testing."""
    return GemmaWithSAE(
        model=mock_model,
        tokenizer=mock_tokenizer,
        sae=mock_sae,
        sae_config=mock_sae_config,
        max_tokens=100,
    )


# =============================================================================
# Test _format_messages
# =============================================================================


class TestFormatMessages:
    """Tests for _format_messages method."""

    def test_formats_human_message(self, wrapper: GemmaWithSAE) -> None:
        """Human messages should have role 'user'."""
        messages = [HumanMessage(content="Hello")]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_formats_system_message(self, wrapper: GemmaWithSAE) -> None:
        """System messages should have role 'system'."""
        messages = [SystemMessage(content="You are helpful")]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"

    def test_formats_ai_message(self, wrapper: GemmaWithSAE) -> None:
        """AI messages should have role 'assistant'."""
        messages = [AIMessage(content="I can help")]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "I can help"

    def test_formats_tool_message(self, wrapper: GemmaWithSAE) -> None:
        """Tool messages should have role 'tool' and include tool_call_id."""
        messages = [ToolMessage(content="Result", tool_call_id="call_123")]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "Result"
        assert result[0]["tool_call_id"] == "call_123"

    def test_formats_ai_message_with_tool_calls(self, wrapper: GemmaWithSAE) -> None:
        """AI messages with tool_calls should include them."""
        tool_calls = [{"id": "call_123", "name": "search", "args": {"q": "test"}}]
        messages = [AIMessage(content="", tool_calls=tool_calls)]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # LangChain normalizes tool_calls, so check structure
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["name"] == "search"

    def test_formats_multiple_messages(self, wrapper: GemmaWithSAE) -> None:
        """Should correctly format a conversation with multiple messages."""
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="User"),
            AIMessage(content="Assistant"),
        ]
        result = wrapper._format_messages(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"


# =============================================================================
# Test _parse_tool_calls
# =============================================================================


class TestParseToolCalls:
    """Tests for _parse_tool_calls method."""

    def test_parse_single_tool_call(self, wrapper: GemmaWithSAE) -> None:
        """Should parse a single tool call."""
        text = '<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>'
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"
        assert tool_calls[0]["args"] == {"query": "test"}
        assert tool_calls[0]["id"].startswith("call_")

    def test_parse_multiple_tool_calls(self, wrapper: GemmaWithSAE) -> None:
        """Should parse multiple tool calls."""
        text = (
            '<tool_call>{"name": "search", "arguments": {"q": "a"}}</tool_call>'
            '<tool_call>{"name": "get", "arguments": {"id": 1}}</tool_call>'
        )
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert content == ""
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "search"
        assert tool_calls[1]["name"] == "get"

    def test_parse_no_tool_calls(self, wrapper: GemmaWithSAE) -> None:
        """Should return empty list when no tool calls present."""
        text = "This is a normal response with no tool calls."
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert content == text
        assert len(tool_calls) == 0

    def test_parse_tool_call_with_surrounding_text(self, wrapper: GemmaWithSAE) -> None:
        """Should extract content and tool calls separately."""
        text = 'Let me search. <tool_call>{"name": "search", "arguments": {}}</tool_call> Done.'
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert "Let me search." in content
        assert "Done." in content
        assert len(tool_calls) == 1

    def test_parse_malformed_json_skipped(self, wrapper: GemmaWithSAE) -> None:
        """Should skip malformed JSON in tool calls."""
        text = "<tool_call>{invalid json}</tool_call>"
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert len(tool_calls) == 0

    def test_parse_tool_call_with_newlines(self, wrapper: GemmaWithSAE) -> None:
        """Should handle tool calls with whitespace/newlines."""
        text = """<tool_call>
        {"name": "search", "arguments": {"query": "test"}}
        </tool_call>"""
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"


# =============================================================================
# Test _generate
# =============================================================================


class TestGenerate:
    """Tests for _generate method."""

    def test_generate_captures_sae(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        mock_sae: MagicMock,
        mock_sae_config: SAEConfig,
    ) -> None:
        """Should always capture SAE activations."""
        wrapper = GemmaWithSAE(
            model=mock_model,
            tokenizer=mock_tokenizer,
            sae=mock_sae,
            sae_config=mock_sae_config,
        )

        mock_result = SAEFeatureResult(
            feature_acts=torch.zeros(10, 1024),
            tokens=["a", "b"],
            answer="SAE response",
            prompt_len=5,
            top_features=torch.zeros(10, 10),
            top_activations=torch.zeros(10, 10),
            l0=50.0,
            fvu=0.1,
        )

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            messages = [HumanMessage(content="Hello")]
            result = wrapper._generate(messages)

            assert result.generations[0].message.content == "SAE response"
            assert wrapper.last_activations is not None
            assert wrapper.last_activations.answer == "SAE response"

    def test_generate_with_tool_calls(
        self,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
        mock_sae: MagicMock,
        mock_sae_config: SAEConfig,
    ) -> None:
        """Should parse and return tool calls in AIMessage."""
        wrapper = GemmaWithSAE(
            model=mock_model,
            tokenizer=mock_tokenizer,
            sae=mock_sae,
            sae_config=mock_sae_config,
        )

        mock_result = SAEFeatureResult(
            feature_acts=torch.zeros(10, 1024),
            tokens=["a"],
            answer='<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>',
            prompt_len=5,
            top_features=torch.zeros(10, 10),
            top_activations=torch.zeros(10, 10),
            l0=50.0,
            fvu=0.1,
        )

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            messages = [HumanMessage(content="Search for test")]
            result = wrapper._generate(messages)

            msg = result.generations[0].message
            assert msg.tool_calls is not None
            assert len(msg.tool_calls) == 1
            assert msg.tool_calls[0]["name"] == "search"


# =============================================================================
# Test bind_tools
# =============================================================================


class TestBindTools:
    """Tests for bind_tools method."""

    def test_bind_tools_stores_schemas(self, wrapper: GemmaWithSAE) -> None:
        """Should convert and store tool schemas."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.convert_to_openai_tool",
            return_value={"type": "function", "function": {"name": "test_tool"}},
        ):
            result = wrapper.bind_tools([mock_tool])

            assert result is wrapper
            assert len(wrapper._bound_tools) == 1

    def test_bind_tools_returns_self(self, wrapper: GemmaWithSAE) -> None:
        """Should return self for chaining."""
        with patch(
            "model_evaluation.main_agent.gemma_wrapper.convert_to_openai_tool",
            return_value={},
        ):
            result = wrapper.bind_tools([])
            assert result is wrapper
