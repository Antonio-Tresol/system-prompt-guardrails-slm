"""Utility functions for extracting and visualizing attention matrices from transformer models."""

import os
import time
from types import TracebackType
from typing import Any, Optional, Type

import numpy as np
import pandas as pd
import plotly.express as px
import psutil
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def extract_attention(
    *,
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> tuple:
    """Generates an answer and run a forward pass on the full sequence to extract attention."""
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 1. Generate the answer first
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7
        )

    # 2. Run one forward pass on the FULL sequence (Prompt + Answer) to get comprehensive attentions
    #    This allows us to see how the generated answer attends back to the prompt.
    with torch.no_grad():
        outputs = model(generated_ids, output_attentions=True)

    # Decode the answer part only for display
    input_len = inputs.input_ids.shape[-1]
    answer_ids = generated_ids[0][input_len:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    # Convert ALL ids to tokens for the axis labels
    all_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])

    return outputs.attentions, all_tokens, answer


def plot_attention_map(
    *,
    attention_matrix: torch.Tensor,
    tokens: list[str],
    title: str = "Attention Map",
) -> None:
    """Plots an interactive heatmap of the attention matrix using Plotly."""
    # Move to cpu and numpy
    attn_data = attention_matrix.float().cpu().numpy()

    # Create interactive heatmap
    fig = px.imshow(
        attn_data,
        x=tokens,
        y=tokens,
        labels={"x": "Key Token", "y": "Query Token", "color": "Attention"},
        title=title,
        color_continuous_scale="Viridis",
        aspect="auto",
    )

    # Improve layout for readability
    fig.update_layout(
        width=800,
        height=800,
        xaxis_tickangle=-45,
    )
    fig.show()


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
            # These might not be available in all torch versions, but they are in newer ones
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

    print(f"{'=' * 60}\n")


class MemoryTracker:
    """Context manager to track memory usage changes during a block of code."""

    def __init__(self, label: str = "Block") -> None:
        self.label = label
        self.start_mem: dict[str, float] = {}
        self.end_mem: dict[str, float] = {}

    def __enter__(self) -> "MemoryTracker":
        self.start_mem = get_memory_usage()
        self.start_time = time.time()
        # Reset max memory tracker for CUDA
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
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

        # Basic RSS delta
        rss_delta = self.end_mem["process_rss_mb"] - self.start_mem["process_rss_mb"]
        print(f"  💾 RSS Change:  {rss_delta:+.2f} MB")

        # CUDA specific details
        if "cuda_allocated_mb" in self.end_mem:
            cuda_delta = self.end_mem["cuda_allocated_mb"] - self.start_mem["cuda_allocated_mb"]
            peak_cuda = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"  🖥️  CUDA Change: {cuda_delta:+.2f} MB")
            print(f"  📈 Peak CUDA:   {peak_cuda:.2f} MB")

        # MPS specific details
        if "mps_allocated_mb" in self.end_mem:
            mps_delta = self.end_mem["mps_allocated_mb"] - self.start_mem["mps_allocated_mb"]
            print(f"  🍎 MPS Change:  {mps_delta:+.2f} MB")

        print("─" * 50 + "\n")


def estimate_kv_cache_size(
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_length: int,
    batch_size: int = 1,
    dtype_size: int = 2,  # 2 for float16/bfloat16, 4 for float32
) -> float:
    """Estimates the KV cache size in MB."""
    # Each token needs 2 vectors (Key and Value) per head per layer
    # Size = num_layers * 2 * num_heads * head_dim * seq_length * batch_size * dtype_size
    cache_bytes = num_layers * 2 * num_heads * head_dim * seq_length * batch_size * dtype_size
    return cache_bytes / (1024 * 1024)


def get_model_memory_footprint(model: PreTrainedModel) -> dict[str, Any]:
    """Calculate the memory footprint of a model."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size

    # Extract model config for KV cache estimation help
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


def calculate_attention_stats(attention_matrix: torch.Tensor) -> dict[str, float]:
    """Calculate statistics for a single attention matrix (seq_len, seq_len)."""
    # attention_matrix is (query_tokens, key_tokens)
    # We focus on the non-masked parts (usually lower triangular for causal)
    # But for a full sequence forward pass, it's fully populated but causal

    attn = attention_matrix.float().cpu().numpy()
    seq_len = attn.shape[0]

    # Entropy: -sum(p * log(p))
    # We calculate entropy for each query token (row) and average it
    # Avoid log(0) by adding epsilon
    epsilon = 1e-12
    row_entropies = -np.sum(attn * np.log(attn + epsilon), axis=1)
    avg_entropy = np.mean(row_entropies)

    # Max possible entropy for this sequence length
    max_entropy = np.log(seq_len)
    normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0

    # Average Distance: sum(attn_ij * |i - j|)
    # Measures how far back the model is looking
    q_idx, k_idx = np.indices((seq_len, seq_len))
    distances = np.abs(q_idx - k_idx)
    avg_distance = np.sum(attn * distances) / seq_len

    # Sparsity: percentage of weights below threshold
    sparsity = np.mean(attn < 0.01)

    return {
        "entropy": avg_entropy,
        "norm_entropy": normalized_entropy,
        "avg_distance": avg_distance,
        "sparsity": sparsity,
    }


def identify_attention_anchors(
    attention_outputs: tuple, tokens: list[str], top_n: int = 5
) -> pd.DataFrame:
    """Identify 'anchor' tokens that receive the most attention across all layers/heads."""
    # attention_outputs is a tuple of (batch, heads, seq, seq)
    seq_len = len(tokens)
    total_received_attention = np.zeros(seq_len)

    for layer_attn in attention_outputs:
        # layer_attn is (batch, heads, q, k)
        # Sum across heads and query tokens to see which key tokens (k) get the most attention
        summed = layer_attn[0].sum(dim=(0, 1)).float().cpu().numpy()
        total_received_attention += summed

    # Normalize by total attention in the system
    total_received_attention /= len(attention_outputs) * attention_outputs[0].shape[1] * seq_len

    anchors = []
    for i in range(seq_len):
        anchors.append({"token": tokens[i], "index": i, "score": total_received_attention[i]})

    df = pd.DataFrame(anchors).sort_values(by="score", ascending=False)
    return df.head(top_n)


def visualize_token_importance(
    attention_outputs: tuple,
    tokens: list[str],
    title: str = "Token Importance (Received Attention)",
) -> None:
    """Visualize which tokens are most 'important' by total received attention."""
    seq_len = len(tokens)
    importance = np.zeros(seq_len)

    for layer_attn in attention_outputs:
        importance += layer_attn[0].sum(dim=(0, 1)).float().cpu().numpy()

    fig = px.bar(
        x=tokens,
        y=importance,
        labels={"x": "Token", "y": "Total Received Attention"},
        title=title,
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()


def analyze_prompt(
    *,
    key: str,
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> None:
    """Enhanced analysis of a prompt."""
    print("\n" + "▓" * 60)
    print(f"🔬 Analyzing Prompt: [{key}]")
    print("▓" * 60)

    with MemoryTracker(f"Inference - {key}"):
        attentions, tokens, answer = extract_attention(
            text=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    print("\n💬 Model Answer:")
    print(f"   {answer}")

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    seq_len = len(tokens)
    print(f"\n📐 Sequence Info:")
    print(f"    Layers: {num_layers} | Heads: {num_heads} | Tokens: {seq_len}")

    # Estimate KV Cache for this sequence
    footprint = get_model_memory_footprint(model)
    kv_size = estimate_kv_cache_size(
        num_layers=footprint["num_layers"],
        num_heads=footprint["num_kv_heads"],
        head_dim=footprint["head_dim"],
        seq_length=seq_len,
    )
    print(f"    KV Cache: {kv_size:.4f} MB")

    # Attention Stats for Last Layer (Average)
    last_layer_attn = attentions[-1][0]
    avg_attn = last_layer_attn.mean(dim=0)
    stats = calculate_attention_stats(avg_attn)
    print("\n📊 Attention Stats (Last Layer, Avg):")
    print(f"    Entropy:      {stats['entropy']:.4f} (Norm: {stats['norm_entropy']:.2f})")
    print(f"    Avg Distance: {stats['avg_distance']:.2f} tokens")
    print(f"    Sparsity:     {stats['sparsity'] * 100:.1f}%")

    # Anchor Tokens
    print("\n🎯 Top Attention Anchors:")
    anchors = identify_attention_anchors(attentions, tokens)
    print(anchors.to_string(index=False))

    print("\n" + "▓" * 60 + "\n")

    # Visualizations
    plot_attention_map(
        attention_matrix=avg_attn,
        tokens=tokens,
        title=f"Avg Attention (Last Layer) - {key}",
    )

    visualize_token_importance(
        attention_outputs=attentions,
        tokens=tokens,
        title=f"Global Token Importance - {key}",
    )
