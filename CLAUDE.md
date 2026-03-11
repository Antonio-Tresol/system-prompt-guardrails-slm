# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An experiment testing whether Markdown-formatted system prompts improve privacy-refusal behaviour in Gemma 3 SLMs (4B and 12B) deployed as RAG agents. The project spans data generation, model evaluation with SAE interpretability, and a LaTeX paper with grounded statistical analysis.

## Commands

```bash
uv sync                          # Install dependencies
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format and sort imports
uv run pytest                    # Run tests (excludes @integration by default)

# CLI entry points
uv run mi_sml_agent              # Interactive chat with the safety agent
uv run run_evaluation            # Run MD vs Plain evaluation pipeline
uv run judge_results --results <path>  # Backfill judge classifications on results

# Paper (see paper/CLAUDE.md for full details)
uv run python -m paper.scripts.run_all                    # Reproduce all statistical results
uv run python paper/validation/validate_claims.py          # Validate paper claims vs raw data
uv run python paper/validation/check_ai_language.py        # Flag AI-style writing in .tex files
cd paper && docker run --rm -v "$(pwd -W):/work" -w //work texlive/texlive latexmk -outdir=out main.tex  # Compile paper (Docker, Windows)
```

## Architecture

### Agent Pipeline

1. Question enters the **Safety Agent** (`model_evaluation/main_agent/rag_agent.py`)
2. Agent uses `think` tool to plan, then `search_knowledge_base` to retrieve context
3. **KB Generator Agent** (`kb_generator/`) dynamically creates 1-3 document chunks per query with `privacy_level` metadata (`public` / `private` / `mixed`)
4. Agent decides: answer (compliance) or refuse (safety) based on privacy labels
5. **GemmaWithSAE** (`gemma_wrapper.py`) captures SAE feature activations for mechanistic analysis
6. **LLM-as-Judge** (`evaluation/judge.py`) classifies responses as refusal/compliance and checks groundedness

### Key Design Decisions

- **Dynamic KB generation** (not static vector store): Prevents data leakage, ensures research integrity
- **`@dynamic_prompt` middleware**: Switches system prompt format (MD/Plain) at runtime for A/B testing
- **Context injection via dataclasses** (`EvaluationContext`): Runtime config without globals
- **Universe contexts as YAML**: Clear separation of public vs private information
- **Named arguments enforced via `*`**: No positional params when multiple parameters exist

### Key Classes

- `GemmaWithSAE` — Custom `BaseChatModel` wrapping Gemma 3 + SAE hooks
- `JumpReLUSAE` — Sparse autoencoder from Gemma Scope 2
- `GeneratorSession` — Stateful KB generation with LangGraph checkpointer
- `EvaluationContext` — Runtime context (privacy flag, session, universe)
- `DocumentChunk` — Structured KB output with privacy metadata

### Data Flow

```
data_generation/universes/*.yaml  -->  KB Generator Agent  -->  Document chunks with privacy labels
data_generation/questions/*.csv   -->  Evaluation Runner    -->  results/{4b,12b}/results.csv + traces/
results/                          -->  Paper scripts        -->  Statistical claims in LaTeX
```

## Project Layout

- `model_evaluation/` — Core agent, evaluation pipeline, SAE integration, tracing
- `data_generation/` — Universe YAMLs, question CSVs, prompt templates (see `data_generation/README.md`)
- `paper/` — LaTeX source, analysis scripts, trace analysis, validation (see `paper/CLAUDE.md`)
- `results/` — Evaluation outputs: `{4b,12b}/results.csv`, `traces/`, `sae_features/`, `kb_cache.json`
- `tests/` — pytest suite mirroring source structure
- `utils/logging.py` — Loguru logging configuration

## Environment Variables

Copy `.env.example` to `.env`. Only two are required:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | API key for OpenRouter |
| `OPENROUTER_BASE_URL` | Yes | `https://openrouter.ai/api/v1` |
| `HF_TOKEN` | No | HuggingFace token (for gated Gemma models) |
| `GEMMA_MODEL_SIZE` | No | `1b`, `4b` (default), `12b`, `27b` |
| `GEMMA_QUANTIZATION` | No | `int4` or None (bf16, default) |
| `KB_GENERATOR_MODEL` | No | OpenRouter model ID (default: `z-ai/glm-4.7`) |
| `JUDGE_MODEL` | No | OpenRouter model ID (default: `openai/gpt-oss-120b`) |
| `MAX_NEW_TOKENS` | No | Max tokens to generate (default: 8000) |

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on PRs and pushes to `main`: ruff lint, ruff format check, pyrefly type check, pytest.

## Code Style

- **Type hints**: All code fully type-hinted, modern syntax (`list[int]` not `List[int]`)
- **Docstrings**: Google-style convention
- **Trailing commas**: Always in multi-line collections
- **Named arguments**: Enforce with `*` when multiple parameters exist
- **Naming**: Descriptive names, no type-encoding prefixes (no `questions_df`), say "model" not "LLM"
- **Comments**: Only when logic isn't self-evident
- **Tests**: pytest, Arrange-Act-Assert, mock external deps, same quality as production code

## Framework Guidelines

- **LangChain / LangGraph**: Always consult latest docs via MCP server when modifying agent code
- **Ruff**: `line-length = 100`, target `py312`, Google docstrings

## Skills

Skills in `.claude/skills/`:

- `/alphaxiv-paper-lookup` — Look up arxiv papers on alphaxiv.org for structured overviews
- `/building-agents-with-modern-langchain` — LangChain/LangGraph agent patterns
- `/gemma-2-scope` — Gemma Scope 2 SAE feature extraction and analysis
- `/convert-py-to-notebook` — Convert Python scripts to Jupyter notebooks
