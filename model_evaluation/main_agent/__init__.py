"""Main agent module for attention analysis on Gemma 3 models.

This package provides modular utilities for:
- Memory tracking and estimation
- Gemma 3 model loading with validation
- Attention extraction and analysis
- SAE feature extraction via Gemma Scope 2
- Visualization

Example:
    from model_evaluation.main_agent import (
        GemmaModelConfig,
        load_gemma_model,
        analyze_prompt_lite,
        load_gemma_scope_sae,
        extract_sae_features,
    )
"""

from model_evaluation.main_agent.attention_extraction import (
    CHAT_TEMPLATE_TOKENS,
    GemmaInterpretabilityResult,
    analyze_prompt,
    analyze_prompt_lite,
    calculate_attention_stats,
    extract_attention,
    identify_attention_anchors,
    is_chat_template_token,
    plot_attention_map,
    visualize_token_importance,
)
from model_evaluation.main_agent.gemma_loader import (
    GemmaModelConfig,
    estimate_kv_cache_memory,
    estimate_model_memory,
    load_gemma_model,
    print_model_info,
    validate_memory_for_context,
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
from model_evaluation.main_agent.gemma_specs import (
    GEMMA_SPECS,
    GemmaSpec,
    get_layer_types,
    get_model_size,
)
from model_evaluation.main_agent.memory import (
    MemoryTracker,
    estimate_attention_matrix_memory,
    estimate_kv_cache_size,
    get_available_memory,
    get_memory_usage,
    get_model_memory_footprint,
    print_memory_usage,
    print_model_memory_footprint,
)

__all__ = [
    # Attention extraction
    "CHAT_TEMPLATE_TOKENS",
    "GemmaInterpretabilityResult",
    "analyze_prompt",
    "analyze_prompt_lite",
    "calculate_attention_stats",
    "extract_attention",
    "identify_attention_anchors",
    "is_chat_template_token",
    "plot_attention_map",
    "visualize_token_importance",
    # Gemma loader
    "GemmaModelConfig",
    "estimate_kv_cache_memory",
    "estimate_model_memory",
    "load_gemma_model",
    "print_model_info",
    "validate_memory_for_context",
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
    # Specs
    "GEMMA_SPECS",
    "GemmaSpec",
    "get_layer_types",
    "get_model_size",
    # Memory
    "MemoryTracker",
    "estimate_attention_matrix_memory",
    "estimate_kv_cache_size",
    "get_available_memory",
    "get_memory_usage",
    "get_model_memory_footprint",
    "print_memory_usage",
    "print_model_memory_footprint",
]
