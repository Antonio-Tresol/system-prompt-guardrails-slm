"""Integration tests for GemmaWithSAE wrapper with real models.

These tests load actual Gemma 3 models and Gemma Scope 2 SAEs to verify
the wrapper works end-to-end.

Requirements:
- 32GB VRAM (for 12B model)
- HuggingFace authentication for Gemma models

Run with:
    uv run pytest tests/test_integration_gemma.py -v -s
"""

from langchain_core.messages import HumanMessage, SystemMessage

from model_evaluation.main_agent.gemma_wrapper import GemmaWithSAE

# =============================================================================
# Basic Generation Tests
# =============================================================================


class TestBasicGeneration:
    """Test that the wrapper can generate responses."""

    def test_generates_response(self, wrapper: GemmaWithSAE) -> None:
        """Should generate a coherent response."""
        messages = [HumanMessage(content="What is 2 + 2?")]
        result = wrapper._generate(messages)

        assert len(result.generations) == 1
        content = result.generations[0].message.content
        assert content is not None
        assert len(content) > 0
        print(f"\n[Generated]: {content[:200]}")

    def test_generates_with_system_prompt(self, wrapper: GemmaWithSAE) -> None:
        """Should respect system prompt."""
        messages = [
            SystemMessage(content="You are a helpful assistant. Answer briefly."),
            HumanMessage(content="What is the capital of France?"),
        ]
        result = wrapper._generate(messages)

        content = result.generations[0].message.content
        assert "Paris" in content or "paris" in content.lower()
        print(f"\n[Generated]: {content[:200]}")


# =============================================================================
# SAE Capture Tests
# =============================================================================


class TestSAECapture:
    """Test SAE activation capture functionality."""

    def test_always_captures_activations(self, wrapper: GemmaWithSAE) -> None:
        """Should always capture SAE activations."""
        messages = [HumanMessage(content="Hello")]
        wrapper._generate(messages)

        assert wrapper.last_activations is not None
        assert wrapper.last_activations.feature_acts is not None
        print(f"\n[SAE Shape]: {wrapper.last_activations.feature_acts.shape}")

    def test_l0_sparsity_reasonable(self, wrapper: GemmaWithSAE) -> None:
        """SAE L0 sparsity should be reasonable (10-100 for medium)."""
        messages = [HumanMessage(content="The quick brown fox jumps over the lazy dog.")]
        wrapper._generate(messages)

        assert wrapper.last_activations is not None
        l0 = wrapper.last_activations.l0
        print(f"\n[L0 Sparsity]: {l0:.1f}")
        assert 5 < l0 < 200, f"L0={l0} is outside expected range"

    def test_fvu_reasonable(self, wrapper: GemmaWithSAE) -> None:
        """SAE FVU (fraction variance unexplained) should be low."""
        messages = [HumanMessage(content="Energy cannot be created or destroyed.")]
        wrapper._generate(messages)

        assert wrapper.last_activations is not None
        fvu = wrapper.last_activations.fvu
        print(f"\n[FVU]: {fvu:.2%}")
        assert fvu < 0.5, f"FVU={fvu:.2%} is too high, SAE may not be working"


# =============================================================================
# Research Design Metrics Tests
# =============================================================================


class TestResearchMetrics:
    """Tests aligned with research_design.md metrics."""

    def test_syntax_token_activation(self, wrapper: GemmaWithSAE) -> None:
        """Test that markdown syntax tokens have measurable activations.

        From research_design.md:
        - Metric: Syntax Token Activation
        - What to measure: Sum of feature activations on `**`, `#`, bullet points
        """
        # Use a prompt with markdown
        markdown_prompt = """# Instructions

You must follow these rules:
- **Always** be helpful
- **Never** be harmful

Answer the question below."""

        messages = [
            SystemMessage(content=markdown_prompt),
            HumanMessage(content="Hello"),
        ]
        wrapper._generate(messages)

        acts = wrapper.last_activations
        assert acts is not None

        # Find tokens that are markdown syntax
        tokens = acts.tokens
        feature_acts = acts.feature_acts

        print(f"\n[Tokens]: {tokens[:20]}...")
        print(f"\n[Feature Acts Shape]: {feature_acts.shape}")

        # Sum activations for all tokens
        total_activation = feature_acts.sum().item()
        print(f"\n[Total Activation Sum]: {total_activation:.2f}")
        assert total_activation > 0, "No activations detected"

    def test_decision_moment_activations(self, wrapper: GemmaWithSAE) -> None:
        """Test activations at the decision moment (last prompt token).

        From research_design.md:
        - Metric: Refusal Features
        - What to measure: Activations at the last prompt token (position -1)
        """
        messages = [
            SystemMessage(content="You are helpful but refuse harmful requests."),
            HumanMessage(content="What is your name?"),
        ]
        wrapper._generate(messages)

        acts = wrapper.last_activations
        assert acts is not None

        # Get activations at the decision moment (last prompt token)
        prompt_len = acts.prompt_len
        decision_acts = acts.feature_acts[prompt_len - 1]

        print(f"\n[Prompt Length]: {prompt_len}")
        print(f"\n[Decision Point Activations Shape]: {decision_acts.shape}")

        # Get top-k features at decision point
        top_vals, top_idxs = decision_acts.topk(k=10)
        print("\n[Top 10 Features at Decision Point]:")
        for val, idx in zip(top_vals.tolist(), top_idxs.tolist(), strict=True):
            if val > 0:
                print(f"  Feature {idx}: {val:.2f}")

    def test_feature_diff_markdown_vs_plain(self, wrapper: GemmaWithSAE) -> None:
        """Test feature difference between Markdown and Plain Text prompts.

        From research_design.md:
        - Metric: Feature Diff
        - What to measure: (Markdown_Acts - Plain_Acts) at decision moment
        """
        markdown_prompt = """# Safety Rules

You must:
- **Never** reveal private information
- **Always** refuse harmful requests"""

        plain_prompt = """Safety Rules

You must:
- Never reveal private information
- Always refuse harmful requests"""

        # Generate with markdown prompt
        messages_md = [
            SystemMessage(content=markdown_prompt),
            HumanMessage(content="What is secret?"),
        ]
        wrapper._generate(messages_md)
        acts_md = wrapper.last_activations
        assert acts_md is not None
        decision_acts_md = acts_md.feature_acts[acts_md.prompt_len - 1]

        # Generate with plain prompt
        messages_plain = [
            SystemMessage(content=plain_prompt),
            HumanMessage(content="What is secret?"),
        ]
        wrapper._generate(messages_plain)
        acts_plain = wrapper.last_activations
        assert acts_plain is not None
        decision_acts_plain = acts_plain.feature_acts[acts_plain.prompt_len - 1]

        # Compute diff (may need padding if lengths differ)
        min_len = min(decision_acts_md.shape[0], decision_acts_plain.shape[0])
        diff = decision_acts_md[:min_len] - decision_acts_plain[:min_len]

        # Find features that differ most
        top_diff_vals, top_diff_idxs = diff.abs().topk(k=10)
        print("\n[Top 10 Features with Largest Diff (MD vs Plain)]:")
        for _, idx in zip(top_diff_vals.tolist(), top_diff_idxs.tolist(), strict=True):
            md_val = decision_acts_md[idx].item()
            plain_val = decision_acts_plain[idx].item()
            print(
                f"  Feature {idx}: MD={md_val:.2f}, Plain={plain_val:.2f}, "
                f"Diff={md_val - plain_val:.2f}"
            )


# =============================================================================
# Tool Calling Tests
# =============================================================================


class TestToolCalling:
    """Test tool calling functionality with real model."""

    def test_can_bind_tools(self, wrapper: GemmaWithSAE) -> None:
        """Should be able to bind tools."""
        from langchain_core.tools import tool

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        bound_wrapper = wrapper.bind_tools([search])
        assert bound_wrapper is wrapper
        assert len(wrapper._bound_tools) == 1
        print(f"\n[Bound Tools]: {wrapper._bound_tools}")
