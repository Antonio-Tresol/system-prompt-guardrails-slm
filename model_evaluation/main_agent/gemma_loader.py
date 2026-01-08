"""Gemma model loading with memory validation and quantization support."""

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from model_evaluation.main_agent.gemma_specs import GEMMA_SPECS, get_layer_types, get_model_size
from model_evaluation.main_agent.memory import (
    estimate_attention_matrix_memory,
    get_available_memory,
)

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GemmaModelConfig:
    """Configuration for loading a Gemma model."""

    model_id: str  # e.g., "google/gemma-3-4b-it"
    quantization: Literal["int4", "int8", None] = None
    max_context_length: int = 32768
    device_map: str = "auto"
    dtype: torch.dtype = torch.bfloat16


# =============================================================================
# Memory Validation
# =============================================================================


def estimate_model_memory(
    model_id: str, quantization: Literal["int4", "int8", None] = None
) -> float:
    """Estimate model weight memory in GB."""
    size = get_model_size(model_id)
    specs = GEMMA_SPECS[size]

    if quantization == "int4":
        return specs["weights_int4_gb"]
    elif quantization == "int8":
        return specs["weights_bf16_gb"] / 2
    else:
        return specs["weights_bf16_gb"]


def estimate_kv_cache_memory(model_id: str, context_length: int) -> float:
    """Estimate KV cache memory in GB."""
    size = get_model_size(model_id)
    specs = GEMMA_SPECS[size]

    num_layers = specs["num_layers"]
    num_kv_heads = specs["num_kv_heads"]
    head_dim = specs["head_dim"]
    window = specs["sliding_window"]
    interval = specs["global_layer_interval"]

    num_global = num_layers // interval
    num_local = num_layers - num_global

    bytes_per_token = 2 * num_kv_heads * head_dim * 2

    global_bytes = num_global * bytes_per_token * context_length
    local_bytes = num_local * bytes_per_token * min(window, context_length)

    return (global_bytes + local_bytes) / (1024**3)


def validate_memory_for_context(
    model_id: str,
    context_length: int,
    quantization: Literal["int4", "int8", None] = None,
) -> tuple[bool, dict[str, float]]:
    """Check if system has enough memory for model + context."""
    size = get_model_size(model_id)
    specs = GEMMA_SPECS[size]

    breakdown = {
        "model_weights_gb": estimate_model_memory(model_id, quantization),
        "kv_cache_gb": estimate_kv_cache_memory(model_id, context_length),
        "attention_global_per_layer_gb": estimate_attention_matrix_memory(
            context_length, specs["num_heads"], "global", specs["sliding_window"]
        ),
        "attention_local_per_layer_gb": estimate_attention_matrix_memory(
            context_length, specs["num_heads"], "local", specs["sliding_window"]
        ),
    }

    min_required = (
        breakdown["model_weights_gb"]
        + breakdown["kv_cache_gb"]
        + breakdown["attention_global_per_layer_gb"]
    )
    breakdown["min_required_gb"] = min_required

    available = get_available_memory()
    if torch.cuda.is_available():
        max_available = max(v for k, v in available.items() if "cuda" in k and "available" in k)
    else:
        max_available = available["ram_available_gb"]

    breakdown["available_gb"] = max_available
    is_safe = max_available >= min_required

    return is_safe, breakdown


# Unsloth pre-quantized model mappings (these work correctly with attention extraction)
UNSLOTH_QUANTIZED_MODELS = {
    "1b": "unsloth/gemma-3-1b-it-bnb-4bit",
    "4b": "unsloth/gemma-3-4b-it-bnb-4bit",
    "12b": "unsloth/gemma-3-12b-it-bnb-4bit",
}


# =============================================================================
# Model Loading
# =============================================================================


def load_gemma_model(
    config: GemmaModelConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a Gemma 3 model with memory validation.

    Uses:
    - Unsloth pre-quantized models for int4 (recommended, works with attention extraction)
    - Gemma3ForCausalLM for text-only inference (all sizes)
    - Native bfloat16 for full precision
    """
    # 1. Validate memory
    is_safe, breakdown = validate_memory_for_context(
        config.model_id,
        config.max_context_length,
        config.quantization,
    )

    if not is_safe:
        raise MemoryError(
            f"Insufficient memory for {config.model_id} at {config.max_context_length} context.\n"
            f"Required: {breakdown['min_required_gb']:.1f} GB\n"
            f"Available: {breakdown['available_gb']:.1f} GB\n"
            f"Tip: Try quantization='int4' or reduce max_context_length."
        )

    # 2. Determine model ID (use Unsloth for int4 quantization)
    size = get_model_size(config.model_id)
    if config.quantization == "int4":
        # Use Unsloth pre-quantized models (faster, no TorchAO/BnB issues)
        actual_model_id = UNSLOTH_QUANTIZED_MODELS.get(size)
        if not actual_model_id:
            raise ValueError(f"No Unsloth int4 model available for size: {size}")
        print(f"📦 Using Unsloth pre-quantized model: {actual_model_id}")
    else:
        actual_model_id = config.model_id

    # 3. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(actual_model_id)

    # 4. Setup load kwargs
    load_kwargs = {
        "device_map": config.device_map,
        "dtype": config.dtype,  # Unsloth models need torch_dtype
        "attn_implementation": "eager",  # Required for attention extraction
    }

    # 5. Load model
    if config.quantization == "int4":
        # Unsloth pre-quantized models must use AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(actual_model_id, **load_kwargs)
    else:
        # Full-precision Google models use Gemma3ForCausalLM
        try:
            from transformers import Gemma3ForCausalLM

            model = Gemma3ForCausalLM.from_pretrained(actual_model_id, **load_kwargs)
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(actual_model_id, **load_kwargs)

    return model, tokenizer


def print_model_info(model_id: str, quantization: Literal["int4", "int8", None] = None) -> None:
    """Print model specifications and memory requirements."""
    size = get_model_size(model_id)
    specs = GEMMA_SPECS[size]
    layer_types = get_layer_types(model_id)

    num_global = sum(1 for t in layer_types.values() if t == "global")
    num_local = sum(1 for t in layer_types.values() if t == "local")

    # Determine display dtype
    dtype_label = (
        "int4" if quantization == "int4" else ("int8" if quantization == "int8" else "bf16")
    )
    weights_gb = estimate_model_memory(model_id, quantization)

    print(f"\n{'═' * 60}")
    print(f"📊 Gemma 3 {size.upper()} Model Info")
    print("═" * 60)
    print("\n🏗️  Architecture:")
    print(f"    Parameters:     {specs['params_b']}B")
    print(f"    Layers:         {specs['num_layers']} ({num_global} global, {num_local} local)")
    print(f"    Heads:          {specs['num_heads']} (KV: {specs['num_kv_heads']})")
    print(f"    Sliding Window: {specs['sliding_window']}")

    print(f"\n💾 Memory ({dtype_label}, 32k context):")
    print(f"    Weights:        {weights_gb:.1f} GB")
    kv_32k = estimate_kv_cache_memory(model_id, 32768)
    print(f"    KV Cache:       {kv_32k:.1f} GB")
    attn_global = estimate_attention_matrix_memory(32768, specs["num_heads"], "global")
    print(f"    Attn (global):  {attn_global:.1f} GB/layer")

    print("═" * 60 + "\n")
