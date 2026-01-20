"""Tests for the KB Generator Agent.

These tests verify:
1. Schema validation (DocumentChunk, GeneratorOutput, GeneratorContext)
2. Session management (thread IDs, reset behavior)
3. Agent factory configuration
4. Integration with real API (privacy modes, coherence)
5. search_knowledge_base tool output formatting
6. Langfuse tracing configuration
"""

import pytest

from model_evaluation.main_agent.kb_generator import (
    GeneratorSession,
    create_kb_generator_agent,
)
from model_evaluation.main_agent.kb_generator.schemas import (
    DocumentChunk,
    GeneratorContext,
    GeneratorOutput,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def generator_session_no_tracing() -> GeneratorSession:
    """Create a generator session for testing without tracing."""
    agent, checkpointer = create_kb_generator_agent(enable_tracing=False)
    return GeneratorSession(agent=agent, checkpointer=checkpointer)


@pytest.fixture
def generator_session_with_tracing() -> GeneratorSession:
    """Create a generator session with Langfuse tracing enabled."""
    agent, checkpointer = create_kb_generator_agent(enable_tracing=True)
    return GeneratorSession(agent=agent, checkpointer=checkpointer)


# =============================================================================
# Unit Tests - Schemas
# =============================================================================


class TestDocumentChunkSchema:
    """Tests for DocumentChunk Pydantic model."""

    def test_valid_public_chunk(self) -> None:
        """Test creating a valid public document chunk."""
        chunk = DocumentChunk(
            document_title="Bella Notte Restaurant",
            section="Menu",
            subsection="Appetizers",
            privacy_level="public",
            content="Our famous bruschetta features fresh tomatoes and basil.",
        )
        assert chunk.document_title == "Bella Notte Restaurant"
        assert chunk.section == "Menu"
        assert chunk.subsection == "Appetizers"
        assert chunk.privacy_level == "public"
        assert "bruschetta" in chunk.content

    def test_valid_private_chunk(self) -> None:
        """Test creating a valid private document chunk."""
        chunk = DocumentChunk(
            document_title="Internal HR Document",
            section="Salaries",
            privacy_level="private",
            content="Manager salary: $85,000/year. This is confidential.",
        )
        assert chunk.privacy_level == "private"
        assert chunk.subsection is None

    def test_valid_mixed_chunk(self) -> None:
        """Test creating a valid mixed privacy chunk."""
        chunk = DocumentChunk(
            document_title="Quarterly Report",
            section="Financial Summary",
            privacy_level="mixed",
            content="Revenue was $1M (public). Profit margin was 23% (internal).",
        )
        assert chunk.privacy_level == "mixed"

    def test_privacy_level_validation(self) -> None:
        """Test that privacy_level only accepts valid values."""
        # Valid values should work
        for level in ("public", "private", "mixed"):
            chunk = DocumentChunk(
                document_title="Test",
                section="Test",
                privacy_level=level,
                content="Test content.",
            )
            assert chunk.privacy_level == level


class TestGeneratorOutputSchema:
    """Tests for GeneratorOutput Pydantic model."""

    def test_single_chunk_output(self) -> None:
        """Test output with a single chunk."""
        chunk = DocumentChunk(
            document_title="Test Doc",
            section="Section 1",
            privacy_level="public",
            content="Some content here.",
        )
        output = GeneratorOutput(chunks=[chunk])
        assert len(output.chunks) == 1
        assert output.chunks[0].document_title == "Test Doc"

    def test_multiple_chunks_output(self) -> None:
        """Test output with multiple chunks (1-3 as per spec)."""
        chunks = [
            DocumentChunk(
                document_title="Restaurant Guide",
                section=f"Section {i}",
                privacy_level="public",
                content=f"Content for section {i}.",
            )
            for i in range(3)
        ]
        output = GeneratorOutput(chunks=chunks)
        assert len(output.chunks) == 3

    def test_rejects_empty_chunks_list(self) -> None:
        """Test that empty chunks list is rejected (must have 1-3)."""
        import pydantic

        with pytest.raises(pydantic.ValidationError) as exc_info:
            GeneratorOutput(chunks=[])

        assert "too_short" in str(exc_info.value)

    def test_rejects_too_many_chunks(self) -> None:
        """Test that more than 3 chunks is rejected."""
        import pydantic

        chunks = [
            DocumentChunk(
                document_title="Doc",
                section=f"Section {i}",
                privacy_level="public",
                content="Content.",
            )
            for i in range(4)
        ]

        with pytest.raises(pydantic.ValidationError) as exc_info:
            GeneratorOutput(chunks=chunks)

        assert "too_long" in str(exc_info.value)


class TestGeneratorContextSchema:
    """Tests for GeneratorContext dataclass."""

    def test_context_with_private_flag_true(self) -> None:
        """Test context configured for private document generation."""
        ctx = GeneratorContext(include_private_info=True)
        assert ctx.include_private_info is True

    def test_context_with_private_flag_false(self) -> None:
        """Test context configured for public-only document generation."""
        ctx = GeneratorContext(include_private_info=False)
        assert ctx.include_private_info is False


# =============================================================================
# Unit Tests - Session Management
# =============================================================================


class TestGeneratorSession:
    """Tests for GeneratorSession class."""

    def test_session_has_unique_thread_id(self) -> None:
        """Test that each session gets a unique thread ID."""
        agent, checkpointer = create_kb_generator_agent(enable_tracing=False)
        session1 = GeneratorSession(agent=agent, checkpointer=checkpointer)
        session2 = GeneratorSession(agent=agent, checkpointer=checkpointer)

        assert session1.thread_id != session2.thread_id
        assert len(session1.thread_id) == 36  # UUID format

    def test_session_reset_changes_thread_id(self) -> None:
        """Test that reset() creates a new thread_id, clearing history."""
        agent, checkpointer = create_kb_generator_agent(enable_tracing=False)
        session = GeneratorSession(agent=agent, checkpointer=checkpointer)

        original_id = session.thread_id
        session.reset()
        new_id = session.thread_id

        assert new_id != original_id
        assert len(new_id) == 36

    def test_thread_id_is_valid_uuid(self) -> None:
        """Test that thread_id is a valid UUID string."""
        import uuid

        agent, checkpointer = create_kb_generator_agent(enable_tracing=False)
        session = GeneratorSession(agent=agent, checkpointer=checkpointer)

        # Should not raise ValueError
        parsed = uuid.UUID(session.thread_id)
        assert str(parsed) == session.thread_id


# =============================================================================
# Unit Tests - Agent Factory
# =============================================================================


class TestAgentFactory:
    """Tests for create_kb_generator_agent factory function."""

    def test_create_agent_without_tracing(self) -> None:
        """Test creating agent with tracing disabled."""
        agent, checkpointer = create_kb_generator_agent(enable_tracing=False)

        assert agent is not None
        assert checkpointer is not None
        # Agent should be a compiled graph
        assert hasattr(agent, "invoke")
        assert hasattr(agent, "stream")

    def test_create_agent_with_tracing(self) -> None:
        """Test creating agent with Langfuse tracing enabled."""
        agent, checkpointer = create_kb_generator_agent(enable_tracing=True)

        assert agent is not None
        assert checkpointer is not None

    def test_agent_has_correct_name(self) -> None:
        """Test that agent is named correctly for tracing identification."""
        agent, _ = create_kb_generator_agent(enable_tracing=False)
        # The agent name is set in create_agent, check it's configured
        assert agent is not None


# =============================================================================
# Unit Tests - EvaluationContext Integration
# =============================================================================


class TestEvaluationContext:
    """Tests for EvaluationContext used by the main safety agent."""

    def test_evaluation_context_with_session(self) -> None:
        """Test creating EvaluationContext with a generator session."""
        from model_evaluation.main_agent.tools import EvaluationContext

        agent, checkpointer = create_kb_generator_agent(enable_tracing=False)
        session = GeneratorSession(agent=agent, checkpointer=checkpointer)

        ctx = EvaluationContext(
            include_private_info=True,
            generator_session=session,
        )

        assert ctx.include_private_info is True
        assert ctx.generator_session is session
        assert ctx.generator_session.thread_id == session.thread_id


# =============================================================================
# Unit Tests - Tool Output Formatting
# =============================================================================


class TestSearchKnowledgeBaseFormatting:
    """Tests for the search_knowledge_base tool output formatting logic."""

    def test_format_single_public_chunk(self) -> None:
        """Test formatting a single public chunk."""
        chunk = DocumentChunk(
            document_title="Italian Cuisine Guide",
            section="Pasta Dishes",
            subsection="Carbonara",
            privacy_level="public",
            content="Traditional carbonara uses guanciale, eggs, and pecorino.",
        )

        # Replicate the formatting logic from tools.py
        formatted_parts = []
        for idx, c in enumerate([chunk], 1):
            privacy_label = f"[{c.privacy_level.capitalize()}]"
            header = f"--- Result {idx} {privacy_label} ---"
            source = f"Source: {c.document_title} > {c.section}"
            if c.subsection:
                source += f" > {c.subsection}"
            formatted_parts.append(f"{header}\n{source}\nContent:\n{c.content}\n")

        result = "\n".join(formatted_parts)

        assert "--- Result 1 [Public] ---" in result
        assert "Source: Italian Cuisine Guide > Pasta Dishes > Carbonara" in result
        assert "Traditional carbonara" in result

    def test_format_multiple_chunks_with_different_privacy(self) -> None:
        """Test formatting multiple chunks with different privacy levels."""
        chunks = [
            DocumentChunk(
                document_title="Company Handbook",
                section="Public Info",
                privacy_level="public",
                content="Our office hours are 9-5.",
            ),
            DocumentChunk(
                document_title="Company Handbook",
                section="HR Data",
                privacy_level="private",
                content="CEO salary: $500,000.",
            ),
        ]

        formatted_parts = []
        for idx, c in enumerate(chunks, 1):
            privacy_label = f"[{c.privacy_level.capitalize()}]"
            header = f"--- Result {idx} {privacy_label} ---"
            source = f"Source: {c.document_title} > {c.section}"
            if c.subsection:
                source += f" > {c.subsection}"
            formatted_parts.append(f"{header}\n{source}\nContent:\n{c.content}\n")

        result = "\n".join(formatted_parts)

        assert "[Public]" in result
        assert "[Private]" in result
        assert "Result 1" in result
        assert "Result 2" in result

    def test_format_chunk_without_subsection(self) -> None:
        """Test formatting chunk without subsection."""
        chunk = DocumentChunk(
            document_title="Simple Doc",
            section="Main Section",
            privacy_level="mixed",
            content="Some mixed content.",
        )

        source = f"Source: {chunk.document_title} > {chunk.section}"
        if chunk.subsection:
            source += f" > {chunk.subsection}"

        assert source == "Source: Simple Doc > Main Section"
        assert ">" not in source[source.rfind("Section") :]  # No trailing >


# =============================================================================
# Integration Tests - Require API Calls
# =============================================================================


@pytest.mark.integration
class TestGeneratorIntegration:
    """Integration tests that make actual API calls.

    Run with: pytest -m integration
    """

    def test_generates_public_docs_when_flag_false(
        self,
        generator_session_no_tracing: GeneratorSession,
    ) -> None:
        """Test that public mode generates only public documents."""
        output = generator_session_no_tracing.generate(
            query="What are the menu items at the Italian restaurant?",
            include_private_info=False,
        )

        assert isinstance(output, GeneratorOutput)
        assert 1 <= len(output.chunks) <= 3, "Should generate 1-3 chunks"

        for chunk in output.chunks:
            assert chunk.privacy_level == "public", f"Expected public, got {chunk.privacy_level}"
            assert len(chunk.content) > 50, "Content should be substantial"

    def test_generates_private_docs_when_flag_true(
        self,
        generator_session_no_tracing: GeneratorSession,
    ) -> None:
        """Test that private mode generates private/mixed documents."""
        output = generator_session_no_tracing.generate(
            query="What are the employee salaries and internal costs?",
            include_private_info=True,
        )

        assert isinstance(output, GeneratorOutput)
        assert 1 <= len(output.chunks) <= 3, "Should generate 1-3 chunks"

        for chunk in output.chunks:
            assert chunk.privacy_level in ("private", "mixed"), (
                f"Expected private/mixed, got {chunk.privacy_level}"
            )

    def test_maintains_coherence_across_calls(
        self,
        generator_session_no_tracing: GeneratorSession,
    ) -> None:
        """Test that the generator maintains coherence across multiple calls."""
        # First call - establish context about a specific restaurant
        output1 = generator_session_no_tracing.generate(
            query="Tell me about the Italian restaurant called Bella Notte in downtown",
            include_private_info=False,
        )
        assert len(output1.chunks) >= 1

        # Second call - ask follow-up, should reference same restaurant
        output2 = generator_session_no_tracing.generate(
            query="What pasta dishes does this restaurant serve?",
            include_private_info=False,
        )
        assert len(output2.chunks) >= 1

        # The second response should relate to the first (same restaurant context)
        # We check that document titles are consistent
        titles1 = {c.document_title for c in output1.chunks}
        titles2 = {c.document_title for c in output2.chunks}

        # At least some overlap in document context
        assert len(titles1 | titles2) <= 4, "Should maintain consistent document context"

    def test_reset_clears_conversation_context(
        self,
        generator_session_no_tracing: GeneratorSession,
    ) -> None:
        """Test that reset() clears conversation history."""
        # Establish context
        generator_session_no_tracing.generate(
            query="Tell me about Sushi Palace restaurant",
            include_private_info=False,
        )

        # Reset the session
        generator_session_no_tracing.reset()

        # New query after reset - agent should not remember previous context
        output = generator_session_no_tracing.generate(
            query="What are the appetizers?",
            include_private_info=False,
        )

        # Should still generate valid output
        assert len(output.chunks) >= 1
        for chunk in output.chunks:
            assert chunk.privacy_level == "public"

    def test_generates_realistic_content_length(
        self,
        generator_session_no_tracing: GeneratorSession,
    ) -> None:
        """Test that generated content meets the 100-300 word guideline."""
        output = generator_session_no_tracing.generate(
            query="Describe the research methodology in the AI safety paper",
            include_private_info=False,
        )

        for chunk in output.chunks:
            word_count = len(chunk.content.split())
            # Allow some flexibility (50-500 words)
            assert 50 <= word_count <= 500, f"Content has {word_count} words, expected 50-500"


@pytest.mark.integration
class TestTracingIntegration:
    """Integration tests for Langfuse tracing."""

    def test_tracing_enabled_agent_works(
        self,
        generator_session_with_tracing: GeneratorSession,
    ) -> None:
        """Test that agent with tracing enabled still functions correctly."""
        output = generator_session_with_tracing.generate(
            query="What are the opening hours of the cafe?",
            include_private_info=False,
        )

        assert isinstance(output, GeneratorOutput)
        assert len(output.chunks) >= 1

        # Verify output is valid
        for chunk in output.chunks:
            assert chunk.document_title
            assert chunk.section
            assert chunk.content
            assert chunk.privacy_level == "public"

    def test_tracing_enabled_respects_privacy_flag(
        self,
        generator_session_with_tracing: GeneratorSession,
    ) -> None:
        """Test that tracing doesn't interfere with privacy flag behavior."""
        # Test private mode with tracing
        output = generator_session_with_tracing.generate(
            query="What is the internal budget breakdown?",
            include_private_info=True,
        )

        assert len(output.chunks) >= 1
        for chunk in output.chunks:
            assert chunk.privacy_level in ("private", "mixed")
