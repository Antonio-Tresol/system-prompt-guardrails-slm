"""Shared fixtures for Gemma integration tests."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_evaluation.main_agent.gemma_scope_sae import load_gemma_scope_sae
from model_evaluation.main_agent.gemma_wrapper import GemmaWithSAE


@pytest.fixture(scope="session")
def gemma_model() -> AutoModelForCausalLM:
    """Load Gemma 3 4B IT model."""
    return AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )  # type: ignore


@pytest.fixture(scope="session")
def gemma_tokenizer() -> AutoTokenizer:
    """Load Gemma 3 tokenizer."""
    return AutoTokenizer.from_pretrained("google/gemma-3-4b-it")


@pytest.fixture(scope="session")
def gemma_sae() -> tuple:
    """Load Gemma Scope 2 SAE."""
    return load_gemma_scope_sae(
        model_size="4b",
        model_type="it",
        layer=29,
        width="16k",
        l0_size="medium",
    )


@pytest.fixture
def wrapper(
    gemma_model: AutoModelForCausalLM,
    gemma_tokenizer: AutoTokenizer,
    gemma_sae: tuple,
) -> GemmaWithSAE:
    """Create GemmaWithSAE wrapper."""
    sae, sae_config = gemma_sae
    return GemmaWithSAE(
        model=gemma_model,
        tokenizer=gemma_tokenizer,
        sae=sae,
        sae_config=sae_config,
        max_tokens=150,
    )
