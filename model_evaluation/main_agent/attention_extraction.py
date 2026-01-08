"""Attention extraction and analysis functions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from model_evaluation.main_agent.memory import (
    MemoryTracker,
    estimate_kv_cache_size,
    get_model_memory_footprint,
)

CHAT_TEMPLATE_TOKENS = frozenset(
    {
        "<start_of_turn>",
        "<end_of_turn>",
        "user",
        "model",
        "<bos>",
        "<eos>",
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
    }
)


def is_chat_template_token(token: str) -> bool:
    """Check if token is part of chat template structure (not content)."""
    return token in CHAT_TEMPLATE_TOKENS or token.strip() in CHAT_TEMPLATE_TOKENS


# =============================================================================
# Gemma 3 Interpretability Result (Type-Safe)
# =============================================================================


@dataclass
class GemmaInterpretabilityResult:
    """Type-safe container for hybrid extraction results from Gemma 3.

    Attributes:
        global_residuals: Layer index -> residual contribution tensor.
                          Used for global layers (O(n) memory vs O(n²) for attention).
        local_attentions: Layer index -> windowed attention tensor.
                          Used for local layers (window 1024 < hidden 2560).
        tokens: List of decoded tokens from the full sequence.
        answer: The generated answer text.
        prompt_len: Number of tokens in the prompt (before generation).
    """

    global_residuals: dict[int, torch.Tensor]
    local_attentions: dict[int, torch.Tensor]
    tokens: list[str]
    answer: str
    prompt_len: int


def extract_attention(
    *,
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> tuple:
    """Generates an answer and run a forward pass on the full sequence to extract attention.

    Returns:
        tuple: (attentions, tokens, answer, prompt_len)
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 1. Generate the answer first
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7
        )  # ty:ignore[call-non-callable]

    # 2. Run one forward pass on the FULL sequence (Prompt + Answer) to get comprehensive attentions
    #    This allows us to see how the generated answer attends back to the prompt.
    with torch.inference_mode():
        outputs = model(generated_ids, output_attentions=True)

    # Decode the answer part only for display
    input_len = inputs.input_ids.shape[-1]
    answer_ids = generated_ids[0][input_len:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    # Convert ALL ids to tokens for the axis labels
    all_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])

    return outputs.attentions, all_tokens, answer, input_len


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
    """Identify 'anchor' tokens that receive the most attention across all layers/heads, excluding BOS/EOS."""
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
    exclude_tokens = {"<|endoftext|>", "<|beginoftext|>", "<bos>", "<eos>", "<s>", "</s>"}

    for i in range(seq_len):
        token = tokens[i]
        # Skip BOS/EOS or equivalent
        if token.strip().lower() in exclude_tokens:
            continue
        anchors.append({"token": token, "index": i, "score": total_received_attention[i]})

    df = pd.DataFrame(anchors).sort_values(by="score", ascending=False)
    return df.head(top_n)


def plot_attention_map(
    *,
    attention_matrix: torch.Tensor,
    tokens: list[str],
    title: str = "Attention Map",
) -> None:
    """Plot an interactive heatmap of the attention matrix using Plotly."""
    attn_data = attention_matrix.float().cpu().numpy()

    fig = px.imshow(
        attn_data,
        x=tokens,
        y=tokens,
        labels={"x": "Key Token", "y": "Query Token", "color": "Attention"},
        title=title,
        color_continuous_scale="Viridis",
        aspect="auto",
    )

    fig.update_layout(
        width=800,
        height=800,
        xaxis_tickangle=-45,
    )
    fig.show()


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
        attentions, tokens, answer, prompt_len = extract_attention(
            text=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    print("\n💬 Model Answer:")
    print("┌" + "─" * 58 + "┐")
    # Indent the answer for clarity
    for line in answer.strip().split("\n"):
        print(f"│ {line}")
    print("└" + "─" * 58 + "┘")

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    seq_len = len(tokens)
    print("\n📐 Sequence Info:")
    print(f"    Layers: {num_layers} | Heads: {num_heads} | Tokens: {seq_len}")
    print(f"    Split:  Prompt ({prompt_len}) | Answer ({seq_len - prompt_len})")

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

    # Prepare marked tokens for visualization to clearly distinguish prompt from answer
    marked_tokens = [
        f"P_{i}:{t}" if i < prompt_len else f"A_{i - prompt_len}:{t}" for i, t in enumerate(tokens)
    ]

    # Visualizations
    plot_attention_map(
        attention_matrix=avg_attn,
        tokens=marked_tokens,
        title=f"Avg Attention (Last Layer) - {key}",
    )

    visualize_token_importance(
        attention_outputs=attentions,
        tokens=marked_tokens,
        title=f"Global Token Importance - {key}",
    )


def analyze_prompt_lite(
    *,
    key: str,
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
    top_n: int = 20,
    exclude_template: bool = True,
) -> pd.DataFrame:
    """Memory-efficient analysis: returns only top N important tokens.

    Args:
        key: Identifier for the prompt.
        prompt: The prompt text.
        model: The model to analyze.
        tokenizer: The tokenizer.
        device: Device string.
        top_n: Number of top tokens to return.
        exclude_template: Whether to exclude chat template tokens.

    Returns:
        DataFrame with top N tokens by attention importance.
    """
    print(f"\n🔬 Analyzing (Lite): [{key}]")

    # Generate and get attention
    attentions, tokens, answer, prompt_len = extract_attention(
        text=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    print(f"💬 Answer: {answer[:100]}...")
    print(f"📐 Tokens: {len(tokens)} (Prompt: {prompt_len}, Answer: {len(tokens) - prompt_len})")

    # Handle None attentions (e.g., SDPA without eager)
    if attentions is None or attentions[0] is None:
        print("⚠️ Attentions not available. Use attn_implementation='eager' when loading model.")
        return pd.DataFrame()

    # Calculate total received attention per token
    seq_len = len(tokens)
    total_attention = np.zeros(seq_len)

    for layer_attn in attentions:
        summed = layer_attn[0].sum(dim=(0, 1)).float().cpu().numpy()
        total_attention += summed

    # Normalize
    total_attention /= len(attentions) * attentions[0].shape[1] * seq_len

    # Build results, optionally filtering template tokens
    results = []
    for i, token in enumerate(tokens):
        if exclude_template and is_chat_template_token(token):
            continue
        position = "prompt" if i < prompt_len else "answer"
        results.append(
            {
                "index": i,
                "token": token,
                "position": position,
                "attention_score": total_attention[i],
            }
        )

    df = pd.DataFrame(results).sort_values("attention_score", ascending=False)
    top_df = df.head(top_n)

    print(f"\n🎯 Top {top_n} Tokens (excluding template: {exclude_template}):")
    print(top_df.to_string(index=False))

    return top_df
