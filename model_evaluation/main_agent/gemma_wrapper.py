"""GemmaWithSAE wrapper for LangChain.

This module defines a custom LangChain BaseChatModel that wraps a Gemma 3 model
and a Gemma Scope 2 SAE. It always captures SAE activations during generation
for mechanistic interpretability analysis.

It supports both Gemma 4B and 12B models (and their quantized variants),
as the SAE configuration is decoupled from the wrapper logic.
"""

import ast
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from model_evaluation.main_agent.gemma_scope_sae import (
    SAEConfig,
    SAEFeatureResult,
    extract_sae_features,
)

logger = logging.getLogger(__name__)


class GemmaWithSAE(BaseChatModel):
    """Custom wrapper for Gemma 3 with SAE feature extraction.

    This model wraps a HuggingFace Gemma 3 model and a JumpReLU SAE from
    Gemma Scope 2. SAE activations are always captured during generation.
    """

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _sae: Any = PrivateAttr()
    _sae_config: SAEConfig = PrivateAttr()
    _bound_tools: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _last_activations: Optional[SAEFeatureResult] = PrivateAttr(default=None)

    model_name: str = "gemma-3-sae"
    max_tokens: int = 512

    def __init__(
        self,
        *,
        model: Any,  # noqa: ANN401
        tokenizer: Any,  # noqa: ANN401
        sae: Any,  # noqa: ANN401
        sae_config: SAEConfig,
        max_tokens: int = 512,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize with loaded model, tokenizer, and SAE.

        Args:
            model: Loaded HuggingFace model (4B, 12B, quantized or full).
            tokenizer: Loaded HuggingFace tokenizer.
            sae: Loaded JumpReLUSAE.
            sae_config: Configuration for the SAE.
            max_tokens: Maximum tokens to generate.
            kwargs: Additional arguments for BaseChatModel.
        """
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._model = model
        self._tokenizer = tokenizer
        self._sae = sae
        self._sae_config = sae_config

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "gemma-3-sae"

    @property
    def last_activations(self) -> Optional[SAEFeatureResult]:
        """Access the most recent SAE activations."""
        return self._last_activations

    def bind_tools(
        self,
        tools: List[BaseTool],
        **kwargs: Any,  # noqa: ANN401
    ) -> "GemmaWithSAE":
        """Bind tools to the model for tool calling.

        Args:
            tools: List of LangChain tools to bind.
            kwargs: Additional arguments (ignored).

        Returns:
            Self with tools bound.
        """
        self._bound_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> ChatResult:
        """Generate a response from messages with SAE capture.

        Args:
            messages: List of LangChain messages.
            stop: Stop sequences (currently unused).
            run_manager: Callback manager for LLM run.
            kwargs: Additional generation arguments.

        Returns:
            ChatResult containing the generated message.
        """
        self._log_debug_messages(messages)

        final_messages = self._inject_tool_prompt(messages)
        formatted = self._format_messages(final_messages)
        prompt = self._apply_chat_template(formatted)

        logger.debug("Formatted prompt length: %d chars", len(prompt))
        if len(prompt) < 1000:
            logger.debug("Prompt preview: %s", prompt)

        result = extract_sae_features(
            model=self._model,
            tokenizer=self._tokenizer,
            sae=self._sae,
            sae_config=self._sae_config,
            text=prompt,
            max_new_tokens=self.max_tokens,
        )
        self._last_activations = result
        output_text = result.answer

        content, tool_calls = self._parse_tool_calls(output_text)

        msg = (
            AIMessage(content=content, tool_calls=tool_calls)
            if tool_calls
            else AIMessage(content=output_text)
        )

        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _log_debug_messages(self, messages: List[BaseMessage]) -> None:
        """Log incoming messages for debugging."""
        logger.debug("=== GemmaWithSAE._generate called ===")
        logger.debug("Messages received: %d", len(messages))
        for i, msg in enumerate(messages):
            content_preview = str(msg.content)[:100] if msg.content else ""
            logger.debug("  [%d] %s: %s", i, type(msg).__name__, content_preview)
        logger.debug("Bound tools: %d", len(self._bound_tools))

    def _inject_tool_prompt(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Inject tool definitions into the system prompt if tools are bound."""
        if not self._bound_tools:
            return list(messages)

        final_messages = list(messages)
        tool_prompt = self._get_tool_definitions_prompt()

        if final_messages and isinstance(final_messages[0], SystemMessage):
            # Context: Append to existing system message to consolidate instructions
            new_content = str(final_messages[0].content) + tool_prompt
            final_messages[0] = SystemMessage(content=new_content)
        else:
            final_messages.insert(0, SystemMessage(content=tool_prompt))

        return final_messages

    def _apply_chat_template(self, formatted_messages: List[Dict[str, Any]]) -> str:
        """Apply the tokenizer's chat template."""
        try:
            return self._tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
                # tools parameter is ignored by this tokenizer, handled manually
            )
        except Exception as e:
            logger.error("Template error! Messages: %s", formatted_messages)
            raise e

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to HuggingFace chat format.

        Handles strict User/Model alternation by merging System messages into User messages.
        Formats Tool messages as User messages with markdown blocks.
        """
        formatted_messages: List[Dict[str, Any]] = []
        system_buffer = ""

        for msg in messages:
            if isinstance(msg, SystemMessage):
                if system_buffer:
                    system_buffer += "\n\n"
                system_buffer += str(msg.content)
                continue

            elif isinstance(msg, HumanMessage):
                content = str(msg.content) if msg.content else ""
                if system_buffer:
                    content = f"{system_buffer}\n\n{content}"
                    system_buffer = ""
                formatted_messages.append({"role": "user", "content": content})

            elif isinstance(msg, AIMessage):
                content = self._format_ai_message_content(msg)
                formatted_messages.append({"role": "model", "content": content})

            elif isinstance(msg, ToolMessage):
                content = self._format_tool_output(msg)
                formatted_messages.append({"role": "user", "content": content})

        if system_buffer:
            formatted_messages.append({"role": "user", "content": system_buffer})

        return formatted_messages

    def _format_ai_message_content(self, msg: AIMessage) -> str:
        """Format AIMessage content, including tool calls as markdown blocks."""
        content = str(msg.content) if msg.content else ""

        if msg.tool_calls:
            tool_blocks = []
            for tool_call in msg.tool_calls:
                block = self._format_single_tool_call(tool_call)
                if block:
                    tool_blocks.append(block)

            if tool_blocks:
                if content:
                    content += "\n\n"
                content += "\n".join(tool_blocks)

        return content

    def _format_single_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Format a single tool call dict into a markdown block."""
        args = tool_call.get("args", {})

        # Ensure args is a dict
        if isinstance(args, str):
            try:
                args = ast.literal_eval(args)
            except Exception:
                logger.debug("Failed to parse args string: %s", args)
        if not isinstance(args, dict):
            args = {}

        func_name = tool_call.get("name", "unknown")
        # Format args as kwargs-style string: arg1='val1', arg2=123
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        return f"```tool_code\n{func_name}({args_str})\n```"

    def _format_tool_output(self, msg: ToolMessage) -> str:
        """Format tool output as a markdown block inside a User message."""
        content = str(msg.content) if msg.content else ""
        return f"```tool_output\n{content}\n```"

    def _parse_tool_calls(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Parse tool calls from Gemma output (markdown `tool_code` blocks)."""
        tool_calls = []
        pattern = r"```tool_code\s+(.*?)\s+```"
        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            code_block = match.group(1).strip()
            tool_call = self._parse_single_tool_code(code_block)
            if tool_call:
                tool_calls.append(tool_call)

        content = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return content, tool_calls

    def _parse_single_tool_code(self, code_block: str) -> Optional[Dict[str, Any]]:
        """Parse a single python-like function call string."""
        try:
            tree = ast.parse(code_block)
            if not tree.body or not isinstance(tree.body[0], ast.Expr):
                return None

            call_node = tree.body[0].value
            if not isinstance(call_node, ast.Call):
                return None

            func_name = "unknown"
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id

            args = {}
            for keyword in call_node.keywords:
                try:
                    args[keyword.arg] = ast.literal_eval(keyword.value)
                except Exception:
                    args[keyword.arg] = str(keyword.value)

            return {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": func_name,
                "args": args,
                "type": "tool_call",
            }
        except Exception as e:
            logger.warning(f"Failed to parse tool code: {code_block}. Error: {e}")
            return None

    def _get_tool_definitions_prompt(self) -> str:
        """Generate the system prompt section listing available tools."""
        if not self._bound_tools:
            return ""

        tool_descs = []
        for tool in self._bound_tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")

            properties = func.get("parameters", {}).get("properties", {})
            param_parts = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_parts.append(f"{param_name}: {param_type}")

            param_str = ", ".join(param_parts)
            tool_descs.append(f'def {name}({param_str}):\n    """{desc}"""')

        tools_block = "\n\n".join(tool_descs)

        return (
            "\n\nYou have access to the following Python functions:\n\n"
            f"{tools_block}\n\n"
            "To use a tool, you MUST output a Python function call wrapped in a "
            "`tool_code` block.\n"
            "Example:\n"
            "```tool_code\n"
            "tool_name(arg1='value')\n"
            "```\n"
        )
