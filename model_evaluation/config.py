"""Configuration module for model evaluation pipeline.

This module uses pydantic-settings to load environment variables from .env file
for OpenRouter, Langfuse, vector database, and model/SAE configuration.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from model_evaluation.main_agent.gemma_scope_sae import RECOMMENDED_LAYERS


class Settings(BaseSettings):
    """Settings for model evaluation pipeline loaded from environment variables.

    Attributes:
        openrouter_api_key: API key for OpenRouter.
        openrouter_base_url: Base URL for OpenRouter API.
        langfuse_secret_key: Secret key for Langfuse.
        langfuse_public_key: Public key for Langfuse.
        langfuse_base_url: Base URL for Langfuse.
        vector_db_path: Path to the ChromaDB vector database.
        embeddings_model: Model to use for embeddings.
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

    # Vector database
    vector_db_path: str = "./data/vector_db/chroma_db"
    embeddings_model: str = "openai/text-embedding-3-small"

    # Gemma model configuration
    gemma_model_size: Literal["1b", "4b", "12b", "27b"] = "4b"
    gemma_model_type: Literal["pt", "it"] = "it"
    gemma_quantization: Literal["int4", "int8"] | None = None
    gemma_max_context_length: int = 8192

    # SAE configuration
    sae_layer: int | None = None  # None = use recommended layer for model size
    sae_width: Literal["16k", "65k", "262k", "1m"] = "16k"
    sae_l0_size: Literal["small", "medium", "big"] = "medium"

    # Generation configuration
    max_new_tokens: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def gemma_model_id(self) -> str:
        """Get the full HuggingFace model ID."""
        return f"google/gemma-3-{self.gemma_model_size}-{self.gemma_model_type}"

    @property
    def effective_sae_layer(self) -> int:
        """Get the SAE layer to use, falling back to recommended if not set."""
        if self.sae_layer is not None:
            return self.sae_layer
        return RECOMMENDED_LAYERS[self.gemma_model_size]
