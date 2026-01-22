"""Gemma model loading utilities.

This module provides utilities for loading Gemma 3 models with optional quantization
and memory tracking for use with SAE feature extraction.

For quantized models, we use Google's Quantization Aware Trained (QAT) checkpoints
which provide better quality than runtime quantization.
"""

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_gemma_model_id(
    *,
    size: Literal["1b", "4b", "12b", "27b"],
    model_type: Literal["pt", "it"] = "it",
) -> str:
    """Get the HuggingFace model ID for a Gemma 3 model.

    Returns the base model ID (bf16).

    Args:
        size: Model size (1b, 4b, 12b, 27b).
        model_type: Model type (pt=pretrained, it=instruction-tuned).

    Returns:
        HuggingFace model ID string.
    """
    return f"google/gemma-3-{size}-{model_type}"


@dataclass
class GemmaModelConfig:
    """Configuration for loading Gemma models."""

    model_id: str
    size: str
    max_context_length: int = 8192
    device_map: str = "auto"
    dtype: torch.dtype = torch.bfloat16
    tokenizer_id: str | None = None


def load_gemma_model(
    config: GemmaModelConfig,
    *,
    token: str | None = None,
) -> tuple[Any, Any]:
    """Load the Gemma model and tokenizer.

    Args:
        config: Configuration for the model.
        token: HuggingFace authentication token.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading model: {config.model_id}")
    print(f"  Max context: {config.max_context_length}")

    # Determine tokenizer ID (default to model_id if not specified)
    tokenizer_id = config.tokenizer_id or config.model_id

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=token)

    model_kwargs = {
        "device_map": config.device_map,
        "dtype": config.dtype,
        "token": token,
        "attn_implementation": "eager",  # Required for hook compatibility
    }

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        **model_kwargs,
    )

    print(f"✅ Model loaded on: {next(model.parameters()).device}")

    return model, tokenizer


def print_model_info(model_id: str, quantization: str | None = None) -> None:
    """Print estimated memory requirements for a model.

    Args:
        model_id: HuggingFace model ID.
        quantization: Quantization type.
    """
    # Rough estimates based on model size
    size_estimates = {
        "1b": {"bf16": 2.0, "int8": 1.0, "int4": 0.5},
        "4b": {"bf16": 8.0, "int8": 4.0, "int4": 2.0},
        "12b": {"bf16": 24.0, "int8": 12.0, "int4": 6.0},
        "27b": {"bf16": 54.0, "int8": 27.0, "int4": 14.0},
    }

    # Extract size from model_id
    model_size = None
    for size in size_estimates:
        if size in model_id.lower():
            model_size = size
            break

    if model_size is None:
        print(f"Unknown model size for: {model_id}")
        return

    quant_key = quantization if quantization else "bf16"
    estimated_gb = size_estimates[model_size].get(quant_key, "Unknown")

    print(f"\n📊 Memory Estimate for {model_id}:")
    print(f"   Quantization: {quant_key}")
    print(f"   Estimated VRAM: ~{estimated_gb} GB")


def print_memory_usage(label: str = "") -> None:
    """Print current GPU memory usage.

    Args:
        label: Optional label for the printout.
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}GPU Memory: {allocated:.2f}GB allocated, "
        f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB peak"
    )


def clear_memory() -> None:
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class MemoryTracker:
    """Context manager for tracking memory usage during operations.

    Example:
        with MemoryTracker("Loading model"):
            model = load_model()
    """

    def __init__(self, label: str = "Operation") -> None:
        """Initialize memory tracker.

        Args:
            label: Label for the operation being tracked.
        """
        self.label = label
        self.start_allocated = 0.0
        self.start_reserved = 0.0

    def __enter__(self) -> "MemoryTracker":
        """Record starting memory state."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_allocated = torch.cuda.memory_allocated() / 1024**3
            self.start_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"⏳ {self.label}...")
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Print memory delta."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated() / 1024**3
            delta = end_allocated - self.start_allocated
            print(f"✅ {self.label} complete (+{delta:.2f}GB, total: {end_allocated:.2f}GB)")
        else:
            print(f"✅ {self.label} complete")


@contextmanager
def memory_efficient_loading() -> Generator[None, None, None]:
    """Context manager for memory-efficient model loading.

    Clears cache before and after loading to minimize fragmentation.
    """
    clear_memory()
    try:
        yield
    finally:
        clear_memory()
