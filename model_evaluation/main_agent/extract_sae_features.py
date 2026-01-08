# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     main_language: python
#     notebook_metadata_filter: -all
# ---

# %% [markdown]
# # Extract SAE Features - Gemma Scope 2
#
# This notebook demonstrates how to use Gemma Scope 2 SAEs for feature extraction:
# 1. Load Gemma 3 model with memory validation
# 2. Load Gemma Scope 2 SAE (directly from HuggingFace)
# 3. Extract sparse feature activations
# 4. Compare features between Markdown vs Plain Text prompts
#
# SAEs decompose dense activations into ~60 sparse, interpretable features
# per token (L0 ≈ 60), providing memory-efficient mechanistic interpretability.

# %% title="Setup and Imports"
import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login

from model_evaluation.main_agent import (
    GemmaModelConfig,
    MemoryTracker,
    compare_feature_activations,
    extract_sae_features,
    load_gemma_model,
    load_gemma_scope_sae,
    print_memory_usage,
    print_model_info,
    visualize_token_activations,
    visualize_top_features_per_token,
)
from model_evaluation.main_agent.example_prompts import (
    get_markdown_prompts,
    get_plain_prompts,
)

load_dotenv()
torch.set_grad_enabled(False)

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

# %% title="1. Check Memory Requirements"
MODEL_ID = "google/gemma-3-12b-it"

print_model_info(MODEL_ID, quantization="int4")

# %% title="2. Load Model"
config = GemmaModelConfig(
    model_id=MODEL_ID,
    quantization="int4",
    max_context_length=8192,
)

print(f"\n{'=' * 60}")
print(f"Loading {config.model_id}")
print(f"Quantization: {config.quantization or 'None (bf16)'}")
print(f"{'=' * 60}")

with MemoryTracker(f"Loading {config.model_id}"):
    model, tokenizer = load_gemma_model(config)
    model.eval()

device = str(next(model.parameters()).device)
print(f"Model loaded on: {device}")

# %% title="3. Load Gemma Scope 2 SAE"
# Load SAE for residual stream at a late layer (layer 41 for 12B model)
# This captures abstract concepts the model has developed
with MemoryTracker("Loading Gemma Scope 2 SAE"):
    sae, sae_config = load_gemma_scope_sae(
        model_size="12b",
        model_type="it",  # instruction-tuned (matches our model)
        layer=41,  # ~85% depth for abstract concepts (available: 12, 24, 31, 41)
        width="16k",  # 16k features (memory efficient)
        l0_size="medium",  # ~60 active features per token
        device=device,
    )

print("\nSAE Info:")
print(f"  Features: {sae_config.d_sae}")
print(f"  Layer: {sae_config.layer}")
print(f"  Width: {sae_config.width}")

# %% title="4. Test SAE on Simple Prompt"
test_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is the capital of France?"}],
    tokenize=False,
    add_generation_prompt=True,
)

with MemoryTracker("SAE Feature Extraction"):
    result = extract_sae_features(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_config=sae_config,
        text=test_prompt,
        max_new_tokens=50,
        top_k=10,
    )

print("\n📊 SAE Feature Extraction Results:")
num_answer_tokens = len(result.tokens) - result.prompt_len
print(f"  Tokens: {len(result.tokens)} (Prompt: {result.prompt_len}, Answer: {num_answer_tokens})")
print(f"  L0 (avg features/token): {result.l0:.1f}")
print(f"  FVU (reconstruction loss): {result.fvu:.2%}")
print(f"  Answer: {result.answer[:100]}...")

print("\n🎯 Top 10 features at last token:")
last_token_idx = len(result.tokens) - 1
top_feats = result.top_features[last_token_idx].tolist()
top_acts = result.top_activations[last_token_idx].tolist()
for i, (feat, act) in enumerate(zip(top_feats, top_acts, strict=True)):
    print(f"  {i + 1:2d}. Feature {feat:6d}: {act:.4f}")

# %% title="5. Visualize Token Activations"
# Color-coded visualization: brighter green = higher activation
visualize_token_activations(result=result, show_prompt=False)

# %% title="5b. Visualize Top Features per Token"
visualize_top_features_per_token(result=result, num_tokens=15, from_end=True)

# %% [markdown]
# ## Compare Markdown vs Plain Text Prompts
#
# Now let's compare feature activations between Markdown-formatted
# and Plain Text system prompts to see if formatting affects
# which concepts the model activates.

# %% title="5. Generate Prompts"
markdown_prompts = get_markdown_prompts(tokenizer)
plain_prompts = get_plain_prompts(tokenizer)

question_key = "safety_pii"  # Test with the safety-related prompt
md_prompt = markdown_prompts[f"md_{question_key}"]
plain_prompt = plain_prompts[f"plain_{question_key}"]

print(f"Markdown prompt tokens: {len(tokenizer.encode(md_prompt))}")
print(f"Plain text prompt tokens: {len(tokenizer.encode(plain_prompt))}")

# %% title="6. Extract Features - Markdown Prompt"
with MemoryTracker("SAE - Markdown"):
    result_md = extract_sae_features(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_config=sae_config,
        text=md_prompt,
        max_new_tokens=100,
        top_k=20,
    )

print("\n📊 Markdown Results:")
print(f"  L0: {result_md.l0:.1f}")
print(f"  FVU: {result_md.fvu:.2%}")
print(f"  Answer: {result_md.answer[:150]}...")

# %% title="7. Extract Features - Plain Text Prompt"
with MemoryTracker("SAE - Plain Text"):
    result_plain = extract_sae_features(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_config=sae_config,
        text=plain_prompt,
        max_new_tokens=100,
        top_k=20,
    )

print("\n📊 Plain Text Results:")
print(f"  L0: {result_plain.l0:.1f}")
print(f"  FVU: {result_plain.fvu:.2%}")
print(f"  Answer: {result_plain.answer[:150]}...")

# %% title="8. Compare Feature Activations"
compare_feature_activations(
    result_a=result_md,
    result_b=result_plain,
    label_a="Markdown",
    label_b="Plain Text",
    top_n=20,
)

# %% title="8b. Visualize Markdown Answer"
print("📝 Markdown prompt - Answer activations:")
visualize_token_activations(result=result_md, show_prompt=False)

# %% title="8c. Visualize Plain Text Answer"
print("📄 Plain Text prompt - Answer activations:")
visualize_token_activations(result=result_plain, show_prompt=False)

# %% title="8d. Top Features per Token - Markdown"
print("📝 Markdown - Top features per token:")
visualize_top_features_per_token(result=result_md, num_tokens=10, from_end=True)

# %% title="8e. Top Features per Token - Plain"
print("📄 Plain Text - Top features per token:")
visualize_top_features_per_token(result=result_plain, num_tokens=10, from_end=True)

# %% title="9. Final Memory State"
print_memory_usage("Final State")

# %% [markdown]
# ## Analysis Notes
#
# The comparison above shows:
# - **L0**: Average number of active features per token (~60 is expected)
# - **FVU**: Fraction of variance unexplained (lower = better SAE reconstruction)
# - **Top Features**: Which abstract concepts the model activates most strongly
# - **Feature Overlap**: How many features are shared vs unique to each format
# - **Token Visualizations**: Brighter green = higher total feature activation
#
# Features unique to one format may indicate format-specific processing patterns.
# For safety research, look for features that correlate with refusal or compliance.
