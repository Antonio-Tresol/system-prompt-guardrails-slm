"""Gemma Scope 2 SAE feature extraction.

This module provides utilities for extracting sparse autoencoder (SAE) features
from Gemma 3 models using Google's Gemma Scope 2 pretrained SAEs.

SAEs decompose dense model activations into sparse, interpretable features,
enabling mechanistic interpretability research with lower memory overhead
than full attention matrix extraction.

This implementation follows the official Gemma Scope 2 tutorial:
https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r

Example:
    from model_evaluation.main_agent.gemma_scope_sae import (
        load_gemma_scope_sae,
        extract_sae_features,
    )

    # Load SAE for residual stream at layer 40
    sae = load_gemma_scope_sae(
        model_size="12b",
        layer=40,
    )

    # Extract features for a prompt
    result = extract_sae_features(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        target_layer=40,
        text="Your prompt here",
    )
"""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import PreTrainedModel, PreTrainedTokenizer


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder as used in Gemma Scope 2.

    A 2-layer neural network with JumpReLU activation that decomposes
    dense model activations into sparse, interpretable features.
    """

    def __init__(self, *, d_in: int, d_sae: int) -> None:
        """Initialize the SAE.

        Args:
            d_in: Input dimension (model hidden size).
            d_sae: SAE dimension (number of features).
        """
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features.

        Args:
            input_acts: Input activations, shape (..., d_in).

        Returns:
            Sparse feature activations, shape (..., d_sae).
        """
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activations.

        Args:
            acts: Sparse feature activations, shape (..., d_sae).

        Returns:
            Reconstructed activations, shape (..., d_in).
        """
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode.

        Args:
            x: Input activations, shape (..., d_in).

        Returns:
            Reconstructed activations, shape (..., d_in).
        """
        acts = self.encode(x)
        return self.decode(acts)


SAE_REPOS = {
    "1b": {
        "pt": "google/gemma-scope-2-1b-pt",
        "it": "google/gemma-scope-2-1b-it",
    },
    "4b": {
        "pt": "google/gemma-scope-2-4b-pt",
        "it": "google/gemma-scope-2-4b-it",
    },
    "12b": {
        "pt": "google/gemma-scope-2-12b-pt",
        "it": "google/gemma-scope-2-12b-it",
    },
    "27b": {
        "pt": "google/gemma-scope-2-27b-pt",
        "it": "google/gemma-scope-2-27b-it",
    },
}


AVAILABLE_LAYERS = {
    "1b": [7, 13, 17, 22],  # 26 layers total
    "4b": [9, 17, 22, 29],  # 34 layers total
    "12b": [12, 24, 31, 41],  # 48 layers total
    "27b": [16, 31, 40, 53],  # 62 layers total
}


RECOMMENDED_LAYERS = {
    "1b": 22,
    "4b": 22,
    "12b": 41,  # Was 40, corrected to 41
    "27b": 53,
}


EVALUATION_LAYERS: dict[str, tuple[int, int]] = {
    "1b": (13, 22),  # middle (~50%), upper (~85%)
    "4b": (17, 29),
    "12b": (24, 41),
    "27b": (31, 53),
}


def get_evaluation_layers(*, model_size: str) -> tuple[int, int]:
    """Return (middle_layer, upper_layer) for evaluation capture.

    Args:
        model_size: Gemma model size (1b, 4b, 12b, 27b).

    Returns:
        Tuple of (middle_layer, upper_layer) indices.
    """
    return EVALUATION_LAYERS[model_size]


@dataclass
class SAEConfig:
    """Configuration for a loaded SAE.

    Attributes:
        model_size: Gemma model size.
        model_type: Model type (pt=pretrained, it=instruction-tuned).
        layer: Target layer index.
        width: SAE width (e.g., "16k", "65k").
        l0_size: Sparsity level.
        d_in: Input dimension.
        d_sae: SAE dimension (number of features).
    """

    model_size: str
    model_type: str
    layer: int
    width: str
    l0_size: str
    d_in: int
    d_sae: int


@dataclass
class SAEFeatureResult:
    """Container for SAE feature extraction results.

    Attributes:
        feature_acts: Sparse feature activations, shape (seq_len, n_features).
        tokens: List of decoded tokens.
        answer: The generated answer text.
        prompt_len: Number of tokens in the prompt.
        top_features: Indices of top-k most active features per position.
        top_activations: Activation values for top-k features.
        l0: Average number of active features per token.
        fvu: Fraction of variance unexplained by reconstruction.
    """

    feature_acts: torch.Tensor
    tokens: list[str]
    answer: str
    prompt_len: int
    top_features: torch.Tensor
    top_activations: torch.Tensor
    l0: float
    fvu: float


@dataclass
class MultiLayerSAEFeatureResult:
    """Container for multi-layer SAE feature extraction results.

    Attributes:
        layer_results: Mapping from layer index to per-layer SAEFeatureResult.
        answer: The generated answer text (shared across all layers).
        tokens: List of decoded tokens (shared across all layers).
        prompt_len: Number of tokens in the prompt.
    """

    layer_results: dict[int, SAEFeatureResult]
    answer: str
    tokens: list[str]
    prompt_len: int


def load_gemma_scope_sae(
    *,
    model_size: Literal["1b", "4b", "12b", "27b"] = "12b",
    model_type: Literal["pt", "it"] = "it",
    layer: int | None = None,
    width: str = "16k",
    l0_size: Literal["small", "medium", "big"] = "medium",
    device: str = "cuda",
) -> tuple[JumpReLUSAE, SAEConfig]:
    """Load a Gemma Scope 2 SAE from HuggingFace.

    Args:
        model_size: Gemma model size (1b, 4b, 12b, 27b).
        model_type: Model type - pretrained (pt) or instruction-tuned (it).
        layer: Target layer. If None, uses recommended layer for model size.
        width: SAE width (16k, 65k, 262k, 1m).
        l0_size: Sparsity level - small (~30), medium (~60), big (~100).
        device: Device to load SAE on.

    Returns:
        Tuple of (loaded SAE, SAE configuration).
    """
    if layer is None:
        layer = RECOMMENDED_LAYERS[model_size]

    repo_id = SAE_REPOS[model_size][model_type]
    filename = f"resid_post/layer_{layer}_width_{width}_l0_{l0_size}/params.safetensors"

    print(f"📦 Loading SAE from: {repo_id}")
    print(f"   File: {filename}")

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )

    params = load_file(path_to_params, device=device)

    d_in, d_sae = params["w_enc"].shape
    sae = JumpReLUSAE(d_in=d_in, d_sae=d_sae)
    sae.load_state_dict(params)
    sae = sae.to(device)
    sae.eval()

    config = SAEConfig(
        model_size=model_size,
        model_type=model_type,
        layer=layer,
        width=width,
        l0_size=l0_size,
        d_in=d_in,
        d_sae=d_sae,
    )

    print(f"✅ SAE loaded: {d_sae} features, layer {layer}")

    return sae, config


def _gather_acts_hook(
    mod: nn.Module,
    inputs: tuple,
    outputs: tuple,
    *,
    cache: dict,
    key: str,
) -> tuple:
    """Hook function to gather activations from a layer."""
    acts = outputs[0] if isinstance(outputs, tuple) else outputs

    # Remove batch dimension if present
    if acts.dim() == 3:
        acts = acts.squeeze(0)

    cache[key] = acts.detach()
    return outputs


def _get_model_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    """Get the decoder layers from the model, handling different architectures."""
    # Gemma 3 multimodal (Gemma3ForConditionalGeneration)
    # Path: model.language_model.model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        # Try lm.layers directly
        if hasattr(lm, "layers"):
            return lm.layers

    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "text_model") and hasattr(inner.text_model, "layers"):
            return inner.text_model.layers
        if hasattr(inner, "layers"):
            return inner.layers

    if hasattr(model, "layers"):
        return model.layers

    # Debug: print available paths
    debug_info = [f"Model type: {type(model).__name__}"]
    if hasattr(model, "model"):
        debug_info.append(f"  model.model type: {type(model.model).__name__}")
        if hasattr(model.model, "language_model"):
            lm = model.model.language_model
            debug_info.append(f"    language_model type: {type(lm).__name__}")
            attrs = [a for a in dir(lm) if not a.startswith("_")][:15]
            debug_info.append(f"    attrs: {attrs}")
    if hasattr(model, "language_model"):
        lm = model.language_model
        debug_info.append(f"  language_model type: {type(lm).__name__}")
        attrs = [a for a in dir(lm) if not a.startswith("_")][:15]
        debug_info.append(f"  attrs: {attrs}")

    raise AttributeError("Cannot find layers in model.\n" + "\n".join(debug_info))


def gather_residual_activations(
    *,
    model: PreTrainedModel,
    target_layer: int,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather residual stream activations at a specific layer.

    Args:
        model: The Gemma model.
        target_layer: Layer index to gather activations from.
        input_ids: Tokenized input, shape (1, seq_len).

    Returns:
        Residual activations, shape (seq_len, d_model).
    """
    cache: dict[str, torch.Tensor] = {}

    layers = _get_model_layers(model)
    handle = layers[target_layer].register_forward_hook(
        partial(_gather_acts_hook, cache=cache, key="resid_post"),
    )

    try:
        with torch.inference_mode():
            _ = model(input_ids)
    finally:
        handle.remove()

    return cache["resid_post"]


def extract_sae_features(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sae: JumpReLUSAE,
    sae_config: SAEConfig,
    text: str,
    max_new_tokens: int = 100,
    top_k: int = 10,
) -> SAEFeatureResult:
    """Extract SAE features for a prompt after generation.

    This function:
    1. Tokenizes the input
    2. Generates a response
    3. Runs a forward pass on the full sequence
    4. Extracts residual stream activations at the SAE's target layer
    5. Encodes activations through the SAE to get sparse features

    Args:
        model: The Gemma model.
        tokenizer: The tokenizer.
        sae: Loaded JumpReLU SAE.
        sae_config: SAE configuration with layer info.
        text: Input text/prompt.
        max_new_tokens: Maximum tokens to generate.
        top_k: Number of top features to track per position.

    Returns:
        SAEFeatureResult containing sparse feature activations and metadata.
    """
    device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )

    prompt_len = inputs.input_ids.shape[-1]
    answer_ids = generated_ids[0][prompt_len:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    all_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])

    residual_acts = gather_residual_activations(
        model=model,
        target_layer=sae_config.layer,
        input_ids=generated_ids,
    )

    with torch.inference_mode():
        feature_acts = sae.encode(residual_acts.to(torch.float32))
        recon = sae.decode(feature_acts)

    l0 = (feature_acts[1:] > 0).float().sum(-1).mean().item()

    mse = torch.mean((recon[1:] - residual_acts[1:].float()) ** 2)
    var = residual_acts[1:].float().var()
    fvu = (mse / var).item() if var > 0 else 0.0

    top_activations, top_features = feature_acts.topk(k=top_k, dim=-1)

    return SAEFeatureResult(
        feature_acts=feature_acts,
        tokens=all_tokens,
        answer=answer,
        prompt_len=prompt_len,
        top_features=top_features,
        top_activations=top_activations,
        l0=l0,
        fvu=fvu,
    )


def get_top_features_summary(
    *,
    result: SAEFeatureResult,
    position_range: tuple[int, int] | None = None,
) -> dict[int, float]:
    """Get aggregated feature importance across positions.

    Args:
        result: SAE extraction result.
        position_range: Optional (start, end) range of positions to analyze.
                        If None, uses all positions.

    Returns:
        Dictionary mapping feature index to total activation.
    """
    if position_range is None:
        acts = result.feature_acts
    else:
        start, end = position_range
        acts = result.feature_acts[start:end]

    feature_sums = acts.sum(dim=0)

    nonzero_mask = feature_sums > 0
    feature_indices = torch.where(nonzero_mask)[0]
    feature_values = feature_sums[nonzero_mask]

    sorted_indices = feature_values.argsort(descending=True)

    return {int(feature_indices[i].item()): float(feature_values[i].item()) for i in sorted_indices}


def compare_feature_activations(
    *,
    result_a: SAEFeatureResult,
    result_b: SAEFeatureResult,
    label_a: str = "A",
    label_b: str = "B",
    top_n: int = 20,
) -> None:
    """Compare top features between two SAE extraction results.

    Args:
        result_a: First result to compare.
        result_b: Second result to compare.
        label_a: Label for first result.
        label_b: Label for second result.
        top_n: Number of top features to show.
    """
    summary_a = get_top_features_summary(result=result_a)
    summary_b = get_top_features_summary(result=result_b)

    print(f"\n{'=' * 60}")
    print(f"Feature Comparison: {label_a} vs {label_b}")
    print("=" * 60)

    print(f"\nL0 (avg features/token): {label_a}={result_a.l0:.1f}, {label_b}={result_b.l0:.1f}")
    print(f"FVU (reconstruction loss): {label_a}={result_a.fvu:.2%}, {label_b}={result_b.fvu:.2%}")

    print(f"\nTop {top_n} features for {label_a}:")
    for i, (feat, val) in enumerate(list(summary_a.items())[:top_n]):
        in_b = "✓" if feat in summary_b else " "
        print(f"  {i + 1:2d}. Feature {feat:6d}: {val:8.2f} [{in_b}]")

    print(f"\nTop {top_n} features for {label_b}:")
    for i, (feat, val) in enumerate(list(summary_b.items())[:top_n]):
        in_a = "✓" if feat in summary_a else " "
        print(f"  {i + 1:2d}. Feature {feat:6d}: {val:8.2f} [{in_a}]")

    unique_a = set(summary_a.keys()) - set(summary_b.keys())
    unique_b = set(summary_b.keys()) - set(summary_a.keys())
    shared = set(summary_a.keys()) & set(summary_b.keys())

    print("\nFeature overlap:")
    print(f"  Unique to {label_a}: {len(unique_a)}")
    print(f"  Unique to {label_b}: {len(unique_b)}")
    print(f"  Shared: {len(shared)}")


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _token_span(token: str, activation: float, max_act: float) -> str:
    """Create an HTML span for a token with activation-based coloring.

    Args:
        token: The token string.
        activation: The activation value for this token.
        max_act: Maximum activation for normalization.

    Returns:
        HTML span string with background color based on activation.
    """
    norm_act = activation / max_act if max_act > 1e-9 else 0.0
    norm_act = max(0.0, min(1.0, norm_act))

    bg_color = f"rgba(0, 200, 0, {norm_act:.3f})"

    display_token = _escape_html(token)
    display_token = display_token.replace("\n", "⏎")
    if display_token.startswith(" "):
        display_token = " " + display_token[1:]

    style = (
        f"background-color: {bg_color}; "
        "padding: 2px 1px; "
        "border-radius: 3px; "
        "font-family: monospace;"
    )
    return f'<span style="{style}" title="act={activation:.3f}">{display_token}</span>'


def visualize_token_activations(
    *,
    result: SAEFeatureResult,
    feature_idx: int | None = None,
    show_prompt: bool = True,
    show_answer: bool = True,
    skip_bos: bool = True,
) -> None:
    """Visualize token activations as HTML with color-coded highlighting.

    Args:
        result: SAE feature extraction result.
        feature_idx: Specific feature to visualize. If None, uses total activation.
        show_prompt: Whether to show prompt tokens.
        show_answer: Whether to show answer tokens.
        skip_bos: Whether to skip the BOS token (index 0). Default True.
    """
    from IPython.display import HTML, display

    tokens = result.tokens
    prompt_len = result.prompt_len

    if feature_idx is not None:
        # Show activations for a specific feature
        acts = result.feature_acts[:, feature_idx].cpu().numpy()
        title = f"Feature {feature_idx} Activations"
    else:
        # Show total activation (sum across features)
        acts = result.feature_acts.sum(dim=-1).cpu().numpy()
        title = "Total Feature Activations"

    start_idx = 1 if skip_bos else 0
    max_act = acts[start_idx:].max() if acts[start_idx:].max() > 0 else 1.0

    html_parts = [f"<h4>{title}</h4>"]
    html_parts.append('<div style="line-height: 2; font-size: 14px;">')

    if show_prompt:
        html_parts.append('<span style="color: #666; font-size: 12px;">Prompt: </span>')
        for i in range(start_idx, prompt_len):
            html_parts.append(_token_span(tokens[i], acts[i], max_act))
        html_parts.append("<br>")

    # Answer tokens
    if show_answer and len(tokens) > prompt_len:
        html_parts.append('<span style="color: #666; font-size: 12px;">Answer: </span>')
        for i in range(prompt_len, len(tokens)):
            html_parts.append(_token_span(tokens[i], acts[i], max_act))

    html_parts.append("</div>")

    html_output = "".join(html_parts)
    display(HTML(html_output))


def visualize_top_features_per_token(
    *,
    result: SAEFeatureResult,
    num_tokens: int = 10,
    from_end: bool = True,
    skip_bos: bool = True,
) -> None:
    """Visualize top features for the last N tokens.

    Args:
        result: SAE feature extraction result.
        num_tokens: Number of tokens to show.
        from_end: If True, show last N tokens; if False, show first N.
        skip_bos: Whether to skip the BOS token (index 0). Default True.
    """
    from IPython.display import HTML, display

    tokens = result.tokens
    top_features = result.top_features.cpu()
    top_activations = result.top_activations.cpu()

    # Skip BOS token (index 0) when not showing from end
    min_idx = 1 if skip_bos else 0

    if from_end:
        start_idx = max(min_idx, len(tokens) - num_tokens)
        end_idx = len(tokens)
    else:
        start_idx = min_idx
        end_idx = min(min_idx + num_tokens, len(tokens))

    html_parts = ["<h4>Top Features per Token</h4>"]
    html_parts.append(
        '<table style="font-family: monospace; font-size: 12px; border-collapse: collapse;">'
    )
    html_parts.append(
        "<tr><th style='padding: 4px; border: 1px solid #ddd;'>Pos</th>"
        "<th style='padding: 4px; border: 1px solid #ddd;'>Token</th>"
        "<th style='padding: 4px; border: 1px solid #ddd;'>Top Features (idx: activation)</th></tr>"
    )

    for i in range(start_idx, end_idx):
        token = _escape_html(tokens[i]).replace("\n", "⏎")
        if token.startswith("▁"):
            token = "░" + token[1:]

        features_str = ", ".join(
            f"{int(top_features[i, j].item())}:{top_activations[i, j].item():.2f}"
            for j in range(min(5, top_features.shape[1]))
        )

        in_prompt = "📝" if i < result.prompt_len else "💬"
        html_parts.append(
            f"<tr><td style='padding: 4px; border: 1px solid #ddd;'>{in_prompt}{i}</td>"
            f"<td style='padding: 4px; border: 1px solid #ddd;'>{token}</td>"
            f"<td style='padding: 4px; border: 1px solid #ddd;'>{features_str}</td></tr>"
        )

    html_parts.append("</table>")

    html_output = "".join(html_parts)
    display(HTML(html_output))


def gather_multi_layer_residual_activations(
    *,
    model: PreTrainedModel,
    target_layers: list[int],
    input_ids: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Gather residual stream activations at multiple layers in one forward pass.

    Args:
        model: The Gemma model.
        target_layers: Layer indices to gather activations from.
        input_ids: Tokenized input, shape (1, seq_len).

    Returns:
        Dictionary mapping layer index to activations tensor (seq_len, d_model).
    """
    cache: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    layers = _get_model_layers(model)
    for layer_idx in target_layers:
        key = f"resid_post_layer_{layer_idx}"
        handle = layers[layer_idx].register_forward_hook(
            partial(_gather_acts_hook, cache=cache, key=key),
        )
        handles.append(handle)

    try:
        with torch.inference_mode():
            _ = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    return {layer_idx: cache[f"resid_post_layer_{layer_idx}"] for layer_idx in target_layers}


def extract_multi_layer_sae_features(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    saes: dict[int, tuple[JumpReLUSAE, SAEConfig]],
    text: str,
    max_new_tokens: int = 100,
    top_k: int = 10,
) -> MultiLayerSAEFeatureResult:
    """Extract SAE features at multiple layers with shared generation.

    Generates text once, runs one forward pass with hooks on all target layers,
    and encodes through each layer's SAE independently.

    Args:
        model: The Gemma model.
        tokenizer: The tokenizer.
        saes: Mapping from layer index to (SAE, SAEConfig) pairs.
        text: Input text/prompt.
        max_new_tokens: Maximum tokens to generate.
        top_k: Number of top features to track per position.

    Returns:
        MultiLayerSAEFeatureResult with per-layer results sharing the same generation.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )

    prompt_len = inputs.input_ids.shape[-1]
    answer_ids = generated_ids[0][prompt_len:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    all_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])

    all_residuals = gather_multi_layer_residual_activations(
        model=model,
        target_layers=list(saes.keys()),
        input_ids=generated_ids,
    )

    layer_results: dict[int, SAEFeatureResult] = {}
    for layer_idx, (sae, _sae_config) in saes.items():
        residual_acts = all_residuals[layer_idx]

        with torch.inference_mode():
            feature_acts = sae.encode(residual_acts.to(torch.float32))
            recon = sae.decode(feature_acts)

        l0 = (feature_acts[1:] > 0).float().sum(-1).mean().item()

        mse = torch.mean((recon[1:] - residual_acts[1:].float()) ** 2)
        var = residual_acts[1:].float().var()
        fvu = (mse / var).item() if var > 0 else 0.0

        top_acts, top_feats = feature_acts.topk(k=top_k, dim=-1)

        layer_results[layer_idx] = SAEFeatureResult(
            feature_acts=feature_acts,
            tokens=all_tokens,
            answer=answer,
            prompt_len=prompt_len,
            top_features=top_feats,
            top_activations=top_acts,
            l0=l0,
            fvu=fvu,
        )

    return MultiLayerSAEFeatureResult(
        layer_results=layer_results,
        answer=answer,
        tokens=all_tokens,
        prompt_len=prompt_len,
    )
