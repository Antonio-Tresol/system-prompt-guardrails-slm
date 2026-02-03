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
    """Tests for _format_messages method.

    Gemma 3 requires strict user/model alternation, so the wrapper:
    - Wraps HumanMessage content in <user_message> tags
    - Merges SystemMessage into the next user turn via <system_prompt> tags
    - Uses role 'model' (not 'assistant') for AIMessage
    - Formats ToolMessage as a 'user' turn with <tool_result> tags
    """

    def test_formats_human_message(self, wrapper: GemmaWithSAE) -> None:
        """Human messages get wrapped in <user_message> tags with role 'user'."""
        messages = [HumanMessage(content="Hello")]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "<user_message>Hello</user_message>"

    def test_formats_system_message_merged_into_user(self, wrapper: GemmaWithSAE) -> None:
        """System message merges into the following user message with XML tags."""
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hi"),
        ]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "<system_prompt>You are helpful</system_prompt>" in result[0]["content"]
        assert "<user_message>Hi</user_message>" in result[0]["content"]

    def test_standalone_system_message(self, wrapper: GemmaWithSAE) -> None:
        """System message without a following user message becomes a user turn."""
        messages = [SystemMessage(content="You are helpful")]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "<system_prompt>You are helpful</system_prompt>"

    def test_formats_ai_message(self, wrapper: GemmaWithSAE) -> None:
        """AI messages use role 'model' for Gemma compatibility."""
        messages = [AIMessage(content="I can help")]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "model"
        assert result[0]["content"] == "I can help"

    def test_formats_tool_message_as_user(self, wrapper: GemmaWithSAE) -> None:
        """Tool messages become 'user' role with <tool_result> wrapping."""
        messages = [ToolMessage(content="Result", tool_call_id="call_123")]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "<tool_result>Result</tool_result>"

    def test_formats_ai_message_with_tool_calls(self, wrapper: GemmaWithSAE) -> None:
        """AI messages with tool_calls render as Python function call XML blocks."""
        tool_calls = [{"id": "call_123", "name": "search", "args": {"q": "test"}}]
        messages = [AIMessage(content="", tool_calls=tool_calls)]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "model"
        assert "<tool_call>search(q='test')</tool_call>" in result[0]["content"]

    def test_formats_multiple_messages_with_alternation(self, wrapper: GemmaWithSAE) -> None:
        """System merges into user, then model follows — strict alternation."""
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="User"),
            AIMessage(content="Assistant"),
        ]
        result = wrapper._format_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "<system_prompt>System</system_prompt>" in result[0]["content"]
        assert "<user_message>User</user_message>" in result[0]["content"]
        assert result[1]["role"] == "model"
        assert result[1]["content"] == "Assistant"

    def test_merges_consecutive_same_role(self, wrapper: GemmaWithSAE) -> None:
        """Consecutive same-role messages get merged to maintain alternation."""
        messages = [
            HumanMessage(content="First"),
            HumanMessage(content="Second"),
        ]
        result = wrapper._format_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "<user_message>First</user_message>" in result[0]["content"]
        assert "<user_message>Second</user_message>" in result[0]["content"]


# =============================================================================
# Test _parse_tool_calls
# =============================================================================


class TestParseToolCalls:
    """Tests for _parse_tool_calls with Python function call format.

    The wrapper expects: <tool_call>func_name(arg="value")</tool_call>
    parsed via AST, not JSON.
    """

    def test_parse_single_tool_call(self, wrapper: GemmaWithSAE) -> None:
        """Should parse a single Python-style tool call."""
        text = '<tool_call>search(query="test")</tool_call>'
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"
        assert tool_calls[0]["args"] == {"query": "test"}
        assert tool_calls[0]["id"].startswith("call_")

    def test_parse_multiple_tool_calls(self, wrapper: GemmaWithSAE) -> None:
        """Should parse multiple tool calls."""
        text = '<tool_call>search(q="a")</tool_call><tool_call>get(id=1)</tool_call>'
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert content == ""
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "search"
        assert tool_calls[0]["args"] == {"q": "a"}
        assert tool_calls[1]["name"] == "get"
        assert tool_calls[1]["args"] == {"id": 1}

    def test_parse_no_tool_calls(self, wrapper: GemmaWithSAE) -> None:
        """Should return empty list when no tool calls present."""
        text = "This is a normal response with no tool calls."
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert content == text
        assert len(tool_calls) == 0

    def test_parse_tool_call_with_surrounding_text(self, wrapper: GemmaWithSAE) -> None:
        """Should extract content and tool calls separately."""
        text = 'Let me search. <tool_call>search(q="test")</tool_call> Done.'
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert "Let me search." in content
        assert "Done." in content
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"

    def test_parse_malformed_call_skipped(self, wrapper: GemmaWithSAE) -> None:
        """Should skip unparseable content inside tool_call tags."""
        text = "<tool_call>not a valid python call!!!</tool_call>"
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert len(tool_calls) == 0

    def test_parse_tool_call_with_newlines(self, wrapper: GemmaWithSAE) -> None:
        """Should handle tool calls with whitespace/newlines."""
        text = """<tool_call>
        search(query="test")
        </tool_call>"""
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"
        assert tool_calls[0]["args"] == {"query": "test"}

    def test_parse_tool_call_with_multiple_args(self, wrapper: GemmaWithSAE) -> None:
        """Should parse calls with multiple keyword arguments."""
        text = '<tool_call>search(query="hello", limit=10, exact=True)</tool_call>'
        content, tool_calls = wrapper._parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["args"] == {"query": "hello", "limit": 10, "exact": True}


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
            answer='<tool_call>search(q="test")</tool_call>',
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
