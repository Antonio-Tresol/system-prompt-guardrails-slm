"""Main agent module for SAE interpretability on Gemma 3 models.

This package provides:
- GemmaWithSAE: LangChain wrapper with SAE feature extraction
- SAE loading and feature extraction via Gemma Scope 2
- Gemma model loading utilities with quantization support

Example:
    from model_evaluation.main_agent import (
        GemmaWithSAE,
        GemmaModelConfig,
        load_gemma_model,
        load_gemma_scope_sae,
        extract_sae_features,
    )
"""

from model_evaluation.main_agent.gemma_model_loader import (
    GemmaModelConfig,
    MemoryTracker,
    clear_memory,
    get_gemma_model_id,
    load_gemma_model,
    memory_efficient_loading,
    print_memory_usage,
    print_model_info,
)
from model_evaluation.main_agent.gemma_scope_sae import (
    JumpReLUSAE,
    SAEConfig,
    SAEFeatureResult,
    compare_feature_activations,
    extract_sae_features,
    gather_residual_activations,
    get_top_features_summary,
    load_gemma_scope_sae,
    visualize_token_activations,
    visualize_top_features_per_token,
)
from model_evaluation.main_agent.gemma_wrapper import GemmaWithSAE

__all__ = [
    # Wrapper
    "GemmaWithSAE",
    # Model loading
    "GemmaModelConfig",
    "MemoryTracker",
    "clear_memory",
    "get_gemma_model_id",
    "load_gemma_model",
    "memory_efficient_loading",
    "print_memory_usage",
    "print_model_info",
    # SAE extraction
    "JumpReLUSAE",
    "SAEConfig",
    "SAEFeatureResult",
    "compare_feature_activations",
    "extract_sae_features",
    "gather_residual_activations",
    "get_top_features_summary",
    "load_gemma_scope_sae",
    "visualize_token_activations",
    "visualize_top_features_per_token",
]
