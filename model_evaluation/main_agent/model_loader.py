"""Gemma model loading utilities with memory validation and quantization support."""

from dataclasses import dataclass
from typing import Literal, TypedDict

import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class GemmaSpec(TypedDict):
    """Type definition for Gemma model specifications."""

    params_b: float
    weights_bf16_gb: float
    weights_int4_gb: float
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_dim: int
    head_dim: int
    sliding_window: int
    global_layer_interval: int
    model_class: str


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
# Model Specifications (from Gemma 3 technical report)
# =============================================================================

# Model sizes and architecture details
GEMMA_SPECS: dict[str, GemmaSpec] = {
    "1b": {
        "params_b": 1.0,
        "weights_bf16_gb": 2.0,
        "weights_int4_gb": 0.7,
        "num_layers": 26,
        "num_heads": 8,
        "num_kv_heads": 4,
        "hidden_dim": 1152,
        "head_dim": 256,
        "sliding_window": 1024,
        "global_layer_interval": 6,  # 1 global for every 5 local
        "model_class": "causal_lm",  # Uses AutoModelForCausalLM
    },
    "4b": {
        "params_b": 4.0,
        "weights_bf16_gb": 8.0,
        "weights_int4_gb": 2.6,
        "num_layers": 34,
        "num_heads": 16,
        "num_kv_heads": 8,
        "hidden_dim": 2560,
        "head_dim": 256,
        "sliding_window": 1024,
        "global_layer_interval": 6,
        "model_class": "conditional_generation",
    },
    "12b": {
        "params_b": 12.0,
        "weights_bf16_gb": 24.0,
        "weights_int4_gb": 8.0,
        "num_layers": 48,
        "num_heads": 16,
        "num_kv_heads": 8,
        "hidden_dim": 3840,
        "head_dim": 256,
        "sliding_window": 1024,
        "global_layer_interval": 6,
        "model_class": "conditional_generation",
    },
}


def _get_model_size(model_id: str) -> str:
    """Extract model size (1b, 4b, 12b) from model ID."""
    model_id_lower = model_id.lower()
    if "1b" in model_id_lower:
        return "1b"
    elif "4b" in model_id_lower:
        return "4b"
    elif "12b" in model_id_lower:
        return "12b"
    elif "27b" in model_id_lower:
        raise ValueError("27B model is not supported due to memory constraints")
    else:
        raise ValueError(f"Could not determine model size from: {model_id}")


# =============================================================================
# Layer Type Detection
# =============================================================================


def get_layer_types(model_id: str) -> dict[int, Literal["global", "local"]]:
    """Return layer index -> type mapping.

    Gemma 3 uses 5 local (sliding window) layers for every 1 global layer.
    Global layers are at indices: 5, 11, 17, 23, 29...
    """
    size = _get_model_size(model_id)
    specs = GEMMA_SPECS[size]
    num_layers = specs["num_layers"]
    interval = specs["global_layer_interval"]

    layer_types = {}
    for i in range(num_layers):
        # Global layers at positions 5, 11, 17... (every 6th, starting at index 5)
        if (i + 1) % interval == 0:
            layer_types[i] = "global"
        else:
            layer_types[i] = "local"

    return layer_types


# =============================================================================
# Memory Estimation
# =============================================================================


def estimate_model_memory(model_id: str, quantization: str | None = None) -> float:
    """Estimate model weight memory in GB."""
    size = _get_model_size(model_id)
    specs = GEMMA_SPECS[size]

    if quantization == "int4":
        return specs["weights_int4_gb"]
    elif quantization == "int8":
        return specs["weights_bf16_gb"] / 2  # Approximate
    else:
        return specs["weights_bf16_gb"]


def estimate_kv_cache_memory(model_id: str, context_length: int) -> float:
    """Estimate KV cache memory in GB.

    Gemma 3 uses sliding window for local layers (only 1024 tokens cached)
    and full context for global layers.
    """
    size = _get_model_size(model_id)
    specs = GEMMA_SPECS[size]

    num_layers = specs["num_layers"]
    num_kv_heads = specs["num_kv_heads"]
    head_dim = specs["head_dim"]
    window = specs["sliding_window"]
    interval = specs["global_layer_interval"]

    # Count global and local layers
    num_global = num_layers // interval
    num_local = num_layers - num_global

    # KV cache: 2 (K+V) * num_heads * head_dim * seq_len * 2 bytes (bf16)
    bytes_per_token = 2 * num_kv_heads * head_dim * 2

    # Global layers cache full context, local layers cache only window
    global_bytes = num_global * bytes_per_token * context_length
    local_bytes = num_local * bytes_per_token * min(window, context_length)

    total_bytes = global_bytes + local_bytes
    return total_bytes / (1024**3)


def estimate_attention_matrix_memory(
    context_length: int,
    num_heads: int,
    layer_type: Literal["global", "local"],
    window_size: int = 1024,
    dtype_bytes: int = 4,  # float32 for attention scores
) -> float:
    """Estimate attention matrix memory in GB for a single layer.

    Global: seq_len × seq_len (QUADRATIC)
    Local:  seq_len × window_size (LINEAR)
    """
    if layer_type == "global":
        elements = context_length * context_length * num_heads
    else:
        elements = context_length * window_size * num_heads

    return (elements * dtype_bytes) / (1024**3)


def get_available_memory() -> dict[str, float]:
    """Get available memory in GB."""
    result = {}

    # System RAM
    vm = psutil.virtual_memory()
    result["ram_available_gb"] = vm.available / (1024**3)
    result["ram_total_gb"] = vm.total / (1024**3)

    # CUDA
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            total = props.total_memory / (1024**3)
            result[f"cuda_{i}_available_gb"] = total - allocated
            result[f"cuda_{i}_total_gb"] = total

    return result


def validate_memory_for_context(
    model_id: str,
    context_length: int,
    quantization: str | None = None,
) -> tuple[bool, dict[str, float]]:
    """Check if system has enough memory for model + context.

    Returns:
        (is_safe, breakdown) where breakdown shows memory requirements.

    Raises:
        MemoryError: If insufficient memory (when called from load_gemma_model).
    """
    size = _get_model_size(model_id)
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

    # Minimum required = weights + KV cache + one global attention layer
    min_required = (
        breakdown["model_weights_gb"]
        + breakdown["kv_cache_gb"]
        + breakdown["attention_global_per_layer_gb"]
    )
    breakdown["min_required_gb"] = min_required

    # Check available memory
    available = get_available_memory()
    if torch.cuda.is_available():
        max_available = max(v for k, v in available.items() if "cuda" in k and "available" in k)
    else:
        max_available = available["ram_available_gb"]

    breakdown["available_gb"] = max_available
    is_safe = max_available >= min_required

    return is_safe, breakdown


# =============================================================================
# Model Loading
# =============================================================================


def load_gemma_model(
    config: GemmaModelConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a Gemma model with memory validation.

    Args:
        config: Model configuration including model_id, quantization, etc.

    Returns:
        (model, tokenizer) tuple.

    Raises:
        MemoryError: If estimated memory exceeds available VRAM/RAM.
        ValueError: If quantization requested but TorchAO not available.
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
            f"Breakdown:\n"
            f"  - Model weights: {breakdown['model_weights_gb']:.1f} GB\n"
            f"  - KV cache: {breakdown['kv_cache_gb']:.1f} GB\n"
            f"  - Attention (global): {breakdown['attention_global_per_layer_gb']:.2f} GB/layer\n"
            f"Tip: Try quantization='int4' or reduce max_context_length."
        )

    # 2. Setup quantization config
    quant_config = None
    if config.quantization:
        if not TORCHAO_AVAILABLE:
            raise ValueError(
                f"Quantization '{config.quantization}' requested but TorchAO not available. "
                "Install with: pip install torchao"
            )
        if config.quantization == "int4":
            quant_config = TorchAoConfig("int4_weight_only", group_size=128)
        elif config.quantization == "int8":
            quant_config = TorchAoConfig("int8_weight_only")

    # 3. Determine model class
    size = _get_model_size(config.model_id)
    specs = GEMMA_SPECS[size]

    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    # 5. Load model
    # Note: attn_implementation="eager" is required for output_attentions=True
    load_kwargs = {
        "device_map": config.device_map,
        "dtype": config.dtype,
        "attn_implementation": "eager",  # Required for attention extraction
    }
    if quant_config:
        load_kwargs["quantization_config"] = quant_config

    if specs["model_class"] == "causal_lm":
        # 1B model (text-only)
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)
    else:
        # 4B+ models (multimodal capable)
        try:
            from transformers import Gemma3ForConditionalGeneration

            model = Gemma3ForConditionalGeneration.from_pretrained(config.model_id, **load_kwargs)
        except ImportError:
            # Fall back to AutoModelForCausalLM if Gemma3 class not available
            print(
                "Warning: Gemma3ForConditionalGeneration not available, using AutoModelForCausalLM"
            )
            model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)

    return model, tokenizer


def print_model_info(model_id: str, quantization: str | None = None) -> None:
    """Print model specifications and memory requirements."""
    size = _get_model_size(model_id)
    specs = GEMMA_SPECS[size]
    layer_types = get_layer_types(model_id)

    num_global = sum(1 for t in layer_types.values() if t == "global")
    num_local = sum(1 for t in layer_types.values() if t == "local")

    print(f"\n{'═' * 60}")
    print(f"📊 Gemma 3 {size.upper()} Model Info")
    print(f"{'═' * 60}")
    print("\n🏗️  Architecture:")
    print(f"    Parameters:     {specs['params_b']}B")
    print(f"    Layers:         {specs['num_layers']} ({num_global} global, {num_local} local)")
    print(f"    Heads:          {specs['num_heads']} (KV: {specs['num_kv_heads']})")
    print(f"    Hidden Dim:     {specs['hidden_dim']}")
    print(f"    Sliding Window: {specs['sliding_window']}")

    print("\n💾 Memory (bf16, 32k context):")
    print(f"    Weights:        {specs['weights_bf16_gb']:.1f} GB")
    kv_32k = estimate_kv_cache_memory(model_id, 32768)
    print(f"    KV Cache:       {kv_32k:.1f} GB")
    attn_global = estimate_attention_matrix_memory(32768, specs["num_heads"], "global")
    print(f"    Attn (global):  {attn_global:.1f} GB/layer")

    if quantization == "int4":
        print("\n⚡ With int4 quantization:")
        print(f"    Weights:        {specs['weights_int4_gb']:.1f} GB")

    print(f"{'═' * 60}\n")
