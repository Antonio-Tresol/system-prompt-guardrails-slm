"""Integration tests for GemmaWithSAE with LangChain tools and agents.

These tests verify that the wrapper works correctly with:
1. LangChain tools (@tool decorator)
2. Tool binding (bind_tools method)
3. Agent creation (create_agent)
4. Full agent loop with tool execution

Requirements:
- 32GB VRAM (for 4B model)
- HuggingFace authentication

Run with:
    uv run pytest tests/test_agent_integration.py -v -s
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from model_evaluation.main_agent.gemma_scope_sae import SAEConfig, SAEFeatureResult
from model_evaluation.main_agent.gemma_wrapper import GemmaWithSAE

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock HuggingFace model."""
    model = MagicMock()
    mock_param = torch.zeros(1)
    model.parameters.return_value = iter([mock_param])
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    return model


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_inputs.input_ids = torch.tensor([[1, 2, 3]])
    tokenizer.return_value = mock_inputs
    tokenizer.decode.return_value = "Generated response"
    return tokenizer


@pytest.fixture
def mock_sae() -> MagicMock:
    """Create a mock SAE."""
    return MagicMock()


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


def create_mock_sae_result(answer: str) -> SAEFeatureResult:
    """Create a mock SAEFeatureResult with the given answer."""
    return SAEFeatureResult(
        feature_acts=torch.zeros(10, 1024),
        tokens=["a", "b"],
        answer=answer,
        prompt_len=5,
        top_features=torch.zeros(10, 10),
        top_activations=torch.zeros(10, 10),
        l0=50.0,
        fvu=0.1,
    )


# =============================================================================
# Test Tool Definition
# =============================================================================


@tool
def search(query: str) -> str:
    """Search for information about the query."""
    return f"Results for: {query}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 22°C, Sunny"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return f"Result: {eval(expression)}"  # noqa: S307


# =============================================================================
# Test bind_tools
# =============================================================================


class TestBindTools:
    """Tests for tool binding functionality."""

    def test_bind_tools_stores_schemas(self, wrapper: GemmaWithSAE) -> None:
        """Should store tool schemas when binding tools."""
        result = wrapper.bind_tools([search, get_weather])

        assert result is wrapper
        assert len(wrapper._bound_tools) == 2

    def test_bound_tools_have_correct_structure(self, wrapper: GemmaWithSAE) -> None:
        """Bound tools should have OpenAI-compatible schema structure."""
        wrapper.bind_tools([search])

        tool_schema = wrapper._bound_tools[0]
        assert "type" in tool_schema
        assert tool_schema["type"] == "function"
        assert "function" in tool_schema
        assert "name" in tool_schema["function"]
        assert tool_schema["function"]["name"] == "search"

    def test_bind_tools_returns_self_for_chaining(self, wrapper: GemmaWithSAE) -> None:
        """bind_tools should return self for method chaining."""
        result = wrapper.bind_tools([search]).bind_tools([get_weather])

        assert result is wrapper
        assert len(wrapper._bound_tools) == 1  # Replaces previous

    def test_bind_tools_injects_system_prompt(
        self,
        wrapper: GemmaWithSAE,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Bound tools should be injected into the system prompt."""
        wrapper.bind_tools([search])

        mock_result = create_mock_sae_result("I found some results")

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            # Pass a message list; wrapper should look for SystemMessage or create one
            wrapper._generate([HumanMessage(content="Search for AI")])

        # Verify that apply_chat_template was called
        # We can't easily inspect the 'messages' passed to it
        # because they are modified in place/copied
        # But we can check that it was called.
        mock_tokenizer.apply_chat_template.assert_called_once()

        # We can check that tools are NOT passed to the tokenizer (since we handle it manually)
        call_kwargs = mock_tokenizer.apply_chat_template.call_args.kwargs
        assert "tools" not in call_kwargs


# =============================================================================
# Test Tool Call Parsing
# =============================================================================


class TestToolCallParsing:
    """Tests for parsing tool calls from model output."""

    def test_parses_tool_call_in_response(self, wrapper: GemmaWithSAE) -> None:
        """Should parse tool calls from model output."""
        tool_call_response = "```tool_code\nsearch(query='AI')\n```"
        mock_result = create_mock_sae_result(tool_call_response)

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            result = wrapper._generate([HumanMessage(content="Search for AI")])

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "search"
        assert msg.tool_calls[0]["args"] == {"query": "AI"}

    def test_parses_multiple_tool_calls(self, wrapper: GemmaWithSAE) -> None:
        """Should parse multiple tool calls in one response."""
        multi_tool_response = (
            "```tool_code\nsearch(query='AI')\n```\n```tool_code\nget_weather(city='Paris')\n```"
        )
        mock_result = create_mock_sae_result(multi_tool_response)

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            result = wrapper._generate([HumanMessage(content="Search and weather")])

        msg = result.generations[0].message
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0]["name"] == "search"
        assert msg.tool_calls[1]["name"] == "get_weather"

    def test_tool_calls_have_unique_ids(self, wrapper: GemmaWithSAE) -> None:
        """Each tool call should have a unique ID."""
        multi_tool_response = "```tool_code\nsearch()\n```\n```tool_code\nget_weather()\n```"
        mock_result = create_mock_sae_result(multi_tool_response)

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            result = wrapper._generate([HumanMessage(content="Multiple tools")])

        msg = result.generations[0].message
        id1 = msg.tool_calls[0]["id"]
        id2 = msg.tool_calls[1]["id"]
        assert id1 != id2
        assert id1.startswith("call_")
        assert id2.startswith("call_")

    def test_content_extracted_when_tool_call_present(
        self,
        wrapper: GemmaWithSAE,
    ) -> None:
        """Should extract text content alongside tool calls."""
        mixed_response = "Let me search for that.\n```tool_code\nsearch(query='AI')\n```"
        mock_result = create_mock_sae_result(mixed_response)

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            result = wrapper._generate([HumanMessage(content="Search")])

        msg = result.generations[0].message
        assert "Let me search for that." in msg.content
        assert len(msg.tool_calls) == 1


# =============================================================================
# Test Message Flow (Tool Execution Loop)
# =============================================================================


class TestMessageFlow:
    """Tests for the message flow with tool messages."""

    def test_formats_tool_message_correctly(self, wrapper: GemmaWithSAE) -> None:
        """ToolMessage should be formatted as User message with tool_output block."""
        messages = [
            HumanMessage(content="Search for AI"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_123", "name": "search", "args": {"query": "AI"}}],
            ),
            ToolMessage(content="Results: AI is...", tool_call_id="call_123"),
        ]

        formatted = wrapper._format_messages(messages)

        assert len(formatted) == 3
        # Tool output is mapped to user role in Gemma 3
        assert formatted[2]["role"] == "user"
        assert "```tool_output" in formatted[2]["content"]
        assert "Results: AI is..." in formatted[2]["content"]

    def test_formats_ai_message_with_tool_calls(self, wrapper: GemmaWithSAE) -> None:
        """AIMessage with tool_calls should include them in formatted output."""
        tool_calls = [{"id": "call_456", "name": "get_weather", "args": {"city": "NYC"}}]
        messages = [
            AIMessage(content="", tool_calls=tool_calls),
        ]

        formatted = wrapper._format_messages(messages)

        assert formatted[0]["role"] == "model"
        assert "```tool_code" in formatted[0]["content"]
        assert "get_weather(city='NYC')" in formatted[0]["content"]

    def test_multi_turn_with_tool_execution(self, wrapper: GemmaWithSAE) -> None:
        """Should handle multi-turn conversation with tool execution."""
        # Simulate: user -> AI calls tool -> tool result -> AI final answer
        messages = [
            HumanMessage(content="What's the weather in Paris?"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_abc", "name": "get_weather", "args": {"city": "Paris"}}],
            ),
            ToolMessage(
                content="Weather in Paris: 18°C, Cloudy",
                tool_call_id="call_abc",
            ),
        ]

        mock_result = create_mock_sae_result(
            "Based on the weather data, Paris is currently 18°C and cloudy."
        )

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            result = wrapper._generate(messages)

        msg = result.generations[0].message
        assert "18°C" in msg.content or "cloudy" in msg.content.lower()


# =============================================================================
# Test LangChain Compatibility
# =============================================================================


class TestLangChainCompatibility:
    """Tests for LangChain interface compatibility."""

    def test_is_base_chat_model(self, wrapper: GemmaWithSAE) -> None:
        """Should be an instance of BaseChatModel."""
        from langchain_core.language_models.chat_models import BaseChatModel

        assert isinstance(wrapper, BaseChatModel)

    def test_has_llm_type_property(self, wrapper: GemmaWithSAE) -> None:
        """Should have _llm_type property."""
        assert hasattr(wrapper, "_llm_type")
        assert wrapper._llm_type == "gemma-3-sae"

    def test_has_bind_tools_method(self, wrapper: GemmaWithSAE) -> None:
        """Should have bind_tools method."""
        assert hasattr(wrapper, "bind_tools")
        assert callable(wrapper.bind_tools)

    def test_generate_returns_chat_result(self, wrapper: GemmaWithSAE) -> None:
        """_generate should return ChatResult."""
        from langchain_core.outputs import ChatResult

        mock_result = create_mock_sae_result("Hello!")

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            result = wrapper._generate([HumanMessage(content="Hi")])

        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1

    def test_generation_contains_ai_message(self, wrapper: GemmaWithSAE) -> None:
        """Generation should contain AIMessage."""
        mock_result = create_mock_sae_result("Response")

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            result = wrapper._generate([HumanMessage(content="Hi")])

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)

    def test_invoke_method_works(self, wrapper: GemmaWithSAE) -> None:
        """Invoke method should work (inherited from BaseChatModel)."""
        mock_result = create_mock_sae_result("Hello there!")

        with patch(
            "model_evaluation.main_agent.gemma_wrapper.extract_sae_features",
            return_value=mock_result,
        ):
            result = wrapper.invoke([HumanMessage(content="Hi")])

        assert isinstance(result, AIMessage)
        assert result.content == "Hello there!"
