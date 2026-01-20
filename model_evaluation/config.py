"""Configuration module for model evaluation pipeline.

This module uses pydantic-settings to load environment variables from .env file
for OpenRouter, Langfuse, vector database, and model/SAE configuration.

Environment variables are loaded via load_dotenv() at module import time to ensure
they are available before any third-party packages (like huggingface_hub) are imported.
"""

from dotenv import load_dotenv

# Load .env at module import - this is the single source of truth for env loading.
# Must happen before any imports that might read env vars at import time.
load_dotenv()

from typing import Literal  # noqa: E402

from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402

from model_evaluation.main_agent.gemma_scope_sae import RECOMMENDED_LAYERS  # noqa: E402


class Settings(BaseSettings):
    """Settings for model evaluation pipeline loaded from environment variables.

    Attributes:
        openrouter_api_key: API key for OpenRouter.
        openrouter_base_url: Base URL for OpenRouter API.
        langfuse_secret_key: Secret key for Langfuse.
        langfuse_public_key: Public key for Langfuse.
        langfuse_base_url: Base URL for Langfuse.
        kb_generator_model: Model ID for the KB generator agent (OpenRouter format).
        gemma_model_size: Gemma model size (1b, 4b, 12b, 27b).
        gemma_model_type: Model type (pt=pretrained, it=instruction-tuned).
        gemma_quantization: Quantization type (int4, int8, or None for bf16).
        gemma_max_context_length: Maximum context length for the model.
        sae_layer: SAE layer to use (None for recommended layer per model size).
        sae_width: SAE width (16k, 65k, 262k, 1m).
        sae_l0_size: SAE sparsity level (small, medium, big).
        max_new_tokens: Maximum tokens to generate.
    """

    # API credentials
    openrouter_api_key: str
    openrouter_base_url: str
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_base_url: str

    # KB Generator configuration
    kb_generator_model: str = "google/gemini-3-flash-preview"

    # Gemma model configuration
    gemma_model_size: Literal["1b", "4b", "12b", "27b"] = "4b"
    gemma_model_type: Literal["pt", "it"] = "it"
    gemma_quantization: Literal["int4"] | None = None  # int4 uses QAT models
    gemma_max_context_length: int = 8192

    # SAE configuration
    sae_layer: int | None = None  # None means auto-detect
    sae_width: str = "16k"
    sae_l0_size: Literal["small", "medium", "big"] = "medium"

    # Generation configuration
    max_new_tokens: int = 8000

    # HuggingFace authentication (for gated models like Gemma)
    hf_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def gemma_model_id(self) -> str:
        """Get the full HuggingFace model ID."""
        # Note: logic duplicated from gemma_model_loader.get_gemma_model_id
        # to ensure config object has valid self-contained state
        if self.gemma_quantization == "int4":
            return f"google/gemma-3-{self.gemma_model_size}-it-qat-q4_0-gguf"
        return f"google/gemma-3-{self.gemma_model_size}-{self.gemma_model_type}"

    @property
    def gemma_tokenizer_id(self) -> str:
        """Get the HuggingFace ID for the tokenizer (always base model)."""
        return f"google/gemma-3-{self.gemma_model_size}-{self.gemma_model_type}"

    @property
    def effective_sae_layer(self) -> int:
        """Get the SAE layer to use, falling back to recommended if not set."""
        if self.sae_layer is not None:
            return self.sae_layer
        return RECOMMENDED_LAYERS[self.gemma_model_size]
