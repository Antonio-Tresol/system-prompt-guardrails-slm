"""GemmaWithSAE wrapper for LangChain.

This module defines a custom LangChain BaseChatModel that wraps a Gemma 3 model
and a Gemma Scope 2 SAE. It enables extracting SAE features during generation
while maintaining compatibility with LangChain agents.

It supports both Gemma 4B and 12B models (and their quantized variants),
as the SAE configuration is decoupled from the wrapper logic.
"""

from typing import Any, List, Optional, Type, cast

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
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from model_evaluation.main_agent.gemma_scope_sae import (
    SAEConfig,
    SAEFeatureResult,
    extract_sae_features,
)


class GemmaWithSAE(BaseChatModel):
    """Custom wrapper for Gemma 3 with SAE feature extraction."""

    # Using PrivateAttr for non-Pydantic fields that shouldn't be validated
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _sae: Any = PrivateAttr()
    _sae_config: SAEConfig = PrivateAttr()
    
    capture_sae: bool = False
    model_name: str = "gemma-3-sae"
    max_tokens: int = 512

    # Store tools for binding
    _bound_tools: List[dict] = PrivateAttr(default_factory=list)
    
    # Store last activation result
    _last_activations: Optional[SAEFeatureResult] = PrivateAttr(default=None)

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        sae: Any,
        sae_config: SAEConfig,
        max_tokens: int = 512,
        **kwargs: Any
    ) -> None:
        """Initialize with loaded model, tokenizer, and SAE.
        
        Args:
            model: loaded HuggingFace model (4B, 12B, quantized or full)
            tokenizer: loaded HuggingFace tokenizer
            sae: loaded JumpReLUSAE
            sae_config: configuration for the SAE
            **kwargs: additional arguments for BaseChatModel
        """
        super().__init__(**kwargs)
        self._model = model
        self._tokenizer = tokenizer
        self._sae = sae
        self._sae_config = sae_config
        self.max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:
        return "gemma-3-sae"
    
    @property
    def last_activations(self) -> Optional[SAEFeatureResult]:
        """Access the most recent SAE activations."""
        return self._last_activations

    def bind_tools(
        self,
        tools: List[BaseTool],
        **kwargs: Any,
    ) -> "GemmaWithSAE":
        """Bind tools to the model.
        
        Since we are wrapping a raw HuggingFace model, we store the tools here.
        The agent is responsible for including tool descriptions in the system prompt.
        """
        self._bound_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response and extract SAE features."""
        
        # 1. Format messages for Gemma
        formatted_messages = []
        for msg in messages:
            role = "user"
            content = msg.content
            # Handle standard LangChain message types
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            elif isinstance(msg, HumanMessage):
                role = "user"
            
            # Construct message dict compatible with HF chat templates
            message_dict = {"role": role, "content": content if content is not None else ""}
            
            # extract tool calls from AIMessage if present
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                 message_dict["tool_calls"] = msg.tool_calls
            
            # extract tool_call_id from ToolMessage if present
            if isinstance(msg, ToolMessage):
                if hasattr(msg, "tool_call_id"):
                    message_dict["tool_call_id"] = msg.tool_call_id
                if hasattr(msg, "name"):
                    message_dict["name"] = msg.name

            formatted_messages.append(message_dict)
        
        # Notes on Tool Formatting:
        # We rely on the caller (LangChain Agent) to have already inserted 
        # tool schemas into the SystemMessage if this is a ReAct or Tool-calling agent.
        # Gemma 3 technically supports tool use tokens, but standard HF chat templates
        # usually handle the prompt structure.
        
        prompt_text = self._tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=self._bound_tools if self._bound_tools else None
        )

        # 2. Run generation with SAE capture
        # extract_sae_features handles the tokenization, generation, and SAE forwarding
        
        # Determine max tokens
        if self.max_tokens:
             max_new_tokens = self.max_tokens

        result = extract_sae_features(
            model=self._model,
            tokenizer=self._tokenizer,
            sae=self._sae,
            sae_config=self._sae_config,
            text=prompt_text,
            max_new_tokens=max_new_tokens,
        )
        
        # Store result if capture is enabled
        if self.capture_sae:
            self._last_activations = result
        else:
            self._last_activations = None

        generated_text = result.answer

        # 3. Create LangChain Result
        # We treat everything as a robust text response.
        # Tool parsing happens in the OutputParser of the Agent.
        
        generation = ChatGeneration(message=AIMessage(content=generated_text))

        return ChatResult(generations=[generation])
