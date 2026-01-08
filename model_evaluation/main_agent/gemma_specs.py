"""Gemma 3 model architecture specifications and layer type detection."""

from typing import Literal, TypedDict


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


# Model sizes and architecture details (from Gemma 3 technical report)
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


def get_model_size(model_id: str) -> str:
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


def get_layer_types(model_id: str) -> dict[int, Literal["global", "local"]]:
    """Return layer index -> type mapping.

    Gemma 3 uses 5 local (sliding window) layers for every 1 global layer.
    Global layers are at indices: 5, 11, 17, 23, 29...
    """
    size = get_model_size(model_id)
    specs = GEMMA_SPECS[size]
    num_layers = specs["num_layers"]
    interval = specs["global_layer_interval"]

    layer_types = {}
    for i in range(num_layers):
        if (i + 1) % interval == 0:
            layer_types[i] = "global"
        else:
            layer_types[i] = "local"

    return layer_types
