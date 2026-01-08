"""Memory tracking and estimation utilities for model inference."""

import os
import time
from types import TracebackType
from typing import Any, Literal, Optional

import psutil
import torch
from transformers import PreTrainedModel

# =============================================================================
# Memory Usage Tracking
# =============================================================================


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage statistics in MB."""
    memory_info = {}

    # System memory (RAM)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_info["process_rss_mb"] = mem_info.rss / (1024 * 1024)
    memory_info["process_vms_mb"] = mem_info.vms / (1024 * 1024)

    # System-wide memory
    virtual_mem = psutil.virtual_memory()
    memory_info["system_total_mb"] = virtual_mem.total / (1024 * 1024)
    memory_info["system_available_mb"] = virtual_mem.available / (1024 * 1024)
    memory_info["system_percent_used"] = virtual_mem.percent

    # CUDA memory (if available)
    if torch.cuda.is_available():
        memory_info["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        memory_info["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        memory_info["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # MPS memory (if available)
    if torch.backends.mps.is_available():
        memory_info["mps_available"] = True
        try:
            memory_info["mps_allocated_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
            memory_info["mps_driver_allocated_mb"] = torch.mps.driver_allocated_memory() / (
                1024 * 1024
            )
        except (AttributeError, RuntimeError):
            pass

    return memory_info


def print_memory_usage(label: str = "") -> None:
    """Print formatted memory usage statistics."""
    mem = get_memory_usage()

    prefix = f"[{label}] " if label else ""
    print("\n" + "=" * 60)
    print(f"{prefix}Memory Usage Report")
    print("=" * 60)

    print("Process Memory:")
    print(f"  RSS (Resident Set Size): {mem['process_rss_mb']:.2f} MB")
    print(f"  VMS (Virtual Memory):    {mem['process_vms_mb']:.2f} MB")

    print("\nSystem Memory:")
    print(f"  Total:     {mem['system_total_mb']:.2f} MB")
    print(f"  Available: {mem['system_available_mb']:.2f} MB")
    print(f"  Used:      {mem['system_percent_used']:.1f}%")

    if "cuda_allocated_mb" in mem:
        print("\nCUDA Memory:")
        print(f"  Allocated: {mem['cuda_allocated_mb']:.2f} MB")
        print(f"  Reserved:  {mem['cuda_reserved_mb']:.2f} MB")
        print(f"  Max Allocated: {mem['cuda_max_allocated_mb']:.2f} MB")

    if "mps_allocated_mb" in mem:
        print("\nMPS Memory:")
        print(f"  Allocated: {mem['mps_allocated_mb']:.2f} MB")
        print(f"  Driver Allocated: {mem['mps_driver_allocated_mb']:.2f} MB")
    elif mem.get("mps_available"):
        print("\nMPS: Available (detailed stats via process memory)")

    print("=" * 60 + "\n")


class MemoryTracker:
    """Context manager to track memory usage changes during a block of code."""

    def __init__(self, label: str = "Block") -> None:
        self.label = label
        self.start_mem: dict[str, float] = {}
        self.end_mem: dict[str, float] = {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> "MemoryTracker":
        self.start_mem = get_memory_usage()
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.end_mem = get_memory_usage()
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        print("\n" + "─" * 50)
        print(f"📊 Memory Tracking: {self.label}")
        print("─" * 50)
        print(f"  ⏱️  Duration:    {duration:.2f}s")

        rss_delta = self.end_mem["process_rss_mb"] - self.start_mem["process_rss_mb"]
        print(f"  💾 RSS Change:  {rss_delta:+.2f} MB")

        if "cuda_allocated_mb" in self.end_mem:
            cuda_delta = self.end_mem["cuda_allocated_mb"] - self.start_mem["cuda_allocated_mb"]
            peak_cuda = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"  🖥️  CUDA Change: {cuda_delta:+.2f} MB")
            print(f"  📈 Peak CUDA:   {peak_cuda:.2f} MB")

        if "mps_allocated_mb" in self.end_mem:
            mps_delta = self.end_mem["mps_allocated_mb"] - self.start_mem["mps_allocated_mb"]
            print(f"  🍎 MPS Change:  {mps_delta:+.2f} MB")

        print("─" * 50 + "\n")


# =============================================================================
# Memory Estimation
# =============================================================================


def get_available_memory() -> dict[str, float]:
    """Get available memory in GB."""
    result = {}

    vm = psutil.virtual_memory()
    result["ram_available_gb"] = vm.available / (1024**3)
    result["ram_total_gb"] = vm.total / (1024**3)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            total = props.total_memory / (1024**3)
            result[f"cuda_{i}_available_gb"] = total - allocated
            result[f"cuda_{i}_total_gb"] = total

    return result


def estimate_kv_cache_size(
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_length: int,
    batch_size: int = 1,
    dtype_size: int = 2,
) -> float:
    """Estimates the KV cache size in MB."""
    cache_bytes = num_layers * 2 * num_heads * head_dim * seq_length * batch_size * dtype_size
    return cache_bytes / (1024 * 1024)


def estimate_attention_matrix_memory(
    context_length: int,
    num_heads: int,
    layer_type: Literal["global", "local"],
    window_size: int = 1024,
    dtype_bytes: int = 4,
) -> float:
    """Estimate attention matrix memory in GB for a single layer."""
    if layer_type == "global":
        elements = context_length * context_length * num_heads
    else:
        elements = context_length * window_size * num_heads
    return (elements * dtype_bytes) / (1024**3)


# =============================================================================
# Model Memory Footprint
# =============================================================================


def get_model_memory_footprint(model: PreTrainedModel) -> dict[str, Any]:
    """Calculate the memory footprint of a model."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size

    config = model.config
    hidden_size = getattr(config, "hidden_size", 0)
    num_attention_heads = getattr(config, "num_attention_heads", 0)
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    num_hidden_layers = getattr(config, "num_hidden_layers", 0)
    head_dim = hidden_size // num_attention_heads if num_attention_heads > 0 else 0

    return {
        "parameters_mb": param_size / (1024 * 1024),
        "buffers_mb": buffer_size / (1024 * 1024),
        "total_mb": total_size / (1024 * 1024),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "num_layers": num_hidden_layers,
        "num_heads": num_attention_heads,
        "num_kv_heads": num_key_value_heads,
        "head_dim": head_dim,
    }


def print_model_memory_footprint(model: PreTrainedModel, model_name: str = "Model") -> None:
    """Print the memory footprint of a model."""
    footprint = get_model_memory_footprint(model)

    print("\n" + "═" * 60)
    print(f"🧠 {model_name}")
    print("═" * 60)
    print("\n📦 Memory Footprint:")
    print(f"    Parameters:        {footprint['parameters_mb']:.2f} MB")
    print(f"    Buffers:           {footprint['buffers_mb']:.2f} MB")
    print(f"    Total:             {footprint['total_mb']:.2f} MB")
    print(f"    Num Parameters:    {footprint['num_parameters']:,}")
    print(f"    Trainable Params:  {footprint['num_trainable_parameters']:,}")
    print("\n🏗️  Architecture:")
    print(f"    Layers:            {footprint['num_layers']}")
    print(f"    Heads (Q/KV):      {footprint['num_heads']} / {footprint['num_kv_heads']}")
    print(f"    Head Dim:          {footprint['head_dim']}")
    print("═" * 60 + "\n")
