# Safety Prompts for SLM

This project tests the hypothesis that **Markdown-formatted system prompts lead to better instruction following** for refusing to reveal private information in RAG small language model based agents.

## Project Overview

See @../research_design.md for the full research design and experiment overview.
See @../pyproject.toml for available dependencies and project configuration.

## Tech Stack

- **Python 3.12+** with `uv` as the package manager
- **LangChain / LangGraph** for agent orchestration
- **Langfuse** for observability and tracing
- **ChromaDB** for vector storage
- **Transformers / PyTorch** for model inference
- **Gemma Scope 2 SAEs** for interpretability analysis
- **Loguru** for centralized logging (`utils/logging.py`)
- **Pydantic Settings** for configuration management

## Project Structure

```
safety-prompts-for-slm/
├── data_generation/            # Synthetic data generation pipeline
│   ├── agents.py              # Agent definitions (simple & deep)
│   ├── config.py              # Pydantic settings (OpenRouter, Langfuse)
│   ├── constants.py           # Model lists, themes, defaults
│   ├── generate_data.py       # CLI: `uv run generate_corpus`
│   ├── internal_prompts.py    # System prompt templates
│   ├── utils.py               # Helper functions
│   ├── prompts/               # Prompt templates
│   ├── synthetic_data/        # Pre-generated cookbooks
│   ├── questions/             # Generated evaluation questions
│   ├── universe_contexts/     # YAML restaurant definitions
│   │   ├── carnelian_table.yaml
│   │   ├── brine_and_riddle.yaml
│   │   ├── moonlit_granary.yaml
│   │   └── velvet_hourglass.yaml
│   └── question_generator/    # Question generation sub-module
│       ├── agent.py           # Question generator agent
│       ├── cli.py             # CLI: `uv run generate_questions`
│       ├── schemas.py         # Pydantic models for questions
│       └── prompts.py
│
├── model_evaluation/           # Model evaluation and RAG agent
│   ├── chat.py                # Terminal chat interface: `uv run mi_sml_agent`
│   ├── config.py              # Settings (Gemma, SAE, API keys)
│   ├── main_agent/            # Core agent implementation
│   │   ├── gemma_wrapper.py   # GemmaWithSAE (BaseChatModel wrapper)
│   │   ├── gemma_scope_sae.py # JumpReLUSAE loading and feature extraction
│   │   ├── gemma_model_loader.py # Model loading with quantization
│   │   ├── rag_agent.py       # Safety evaluation agent (MD/Plain prompts)
│   │   ├── tools.py           # search_knowledge_base & think tools
│   │   └── kb_generator/      # Dynamic knowledge base generation
│   │       ├── agent.py       # KB generator agent
│   │       ├── schemas.py     # DocumentChunk, GeneratorOutput models
│   │       ├── session.py     # GeneratorSession (stateful generation)
│   │       └── tools.py       # Agent tools
│   └── evaluation/            # Evaluation pipeline
│       ├── cli.py             # CLI: `uv run run_evaluation`
│       ├── runner.py          # Core evaluation loop
│       ├── schemas.py         # QuestionRow, RunResult models
│       ├── kb_cache.py        # Deterministic KB caching
│       └── analysis.ipynb     # Results analysis notebook
│
├── utils/                     # Shared utilities
│   └── logging.py             # Loguru logging configuration
│
├── tests/                     # Test suite (pytest)
│   ├── gemma/                 # GemmaWithSAE and agent tests
│   │   ├── conftest.py        # Shared fixtures
│   │   ├── test_gemma_wrapper.py
│   │   ├── test_wrapper_logic.py
│   │   ├── test_integration_gemma.py
│   │   ├── test_real_agent.py
│   │   ├── test_agent_integration.py
│   │   └── test_middleware.py
│   ├── knowledge_base/        # KB ingestion tests
│   ├── model_evaluation/      # KB generator tests
│   ├── test_chat_features.py
│   └── test_generator_integration.py
│
├── .env.example               # Environment variables template
├── pyproject.toml             # Dependencies, ruff/pytest config
├── research_design.md         # Full research design
└── AGENTS.md                  # Shared project standards (mirrors .claude/CLAUDE.md)
```

## Common Commands

```bash
# Install dependencies
uv sync

# Run linting with auto-fix
uv run ruff check --fix .

# Auto-format and sort imports
uv run ruff format .

# Run tests
uv run pytest
```

## CLI Entry Points

| Command | Module | Purpose |
|---------|--------|---------|
| `uv run generate_corpus` | `data_generation.generate_data:main` | Generate synthetic restaurant cookbooks |
| `uv run generate_questions` | `data_generation.question_generator.cli:main` | Generate evaluation questions from universe contexts |
| `uv run mi_sml_agent` | `model_evaluation.chat:main` | Interactive terminal chat with the safety agent |
| `uv run run_evaluation` | `model_evaluation.evaluation.cli:main` | Run MD vs Plain evaluation pipeline |
| `uv run langgraph dev` | LangGraph CLI | Launch LangGraph Studio for visual debugging |

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | API key for OpenRouter |
| `OPENROUTER_BASE_URL` | Yes | `https://openrouter.ai/api/v1` |
| `LANGFUSE_SECRET_KEY` | Yes | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | Yes | Langfuse public key |
| `LANGFUSE_BASE_URL` | Yes | `https://us.cloud.langfuse.com` |
| `LANGSMITH_API_KEY` | No | LangSmith API key (for Studio debugging) |
| `HF_TOKEN` | No | HuggingFace token (for gated Gemma models) |
| `GEMMA_MODEL_SIZE` | No | `1b`, `4b` (default), `12b`, `27b` |
| `GEMMA_MODEL_TYPE` | No | `pt` (pretrained) or `it` (instruction-tuned, default) |
| `GEMMA_QUANTIZATION` | No | `int4` or None (bf16, default) |
| `SAE_WIDTH` | No | `16k` (default), `65k`, `262k`, `1m` |
| `SAE_L0_SIZE` | No | `small`, `medium` (default), `big` |
| `KB_GENERATOR_MODEL` | No | OpenRouter model ID for KB generation |
| `MAX_NEW_TOKENS` | No | Max tokens to generate (default: 8000) |

## Architecture

### Agent Pipeline

1. User question enters the **Safety Agent** (`rag_agent.py`)
2. Agent uses `think` tool to plan, then `search_knowledge_base` to retrieve context
3. **KB Generator Agent** dynamically creates 1-3 document chunks per query (not a static vector store)
4. Each chunk has a `privacy_level` (`public` / `private` / `mixed`)
5. Agent decides: answer (compliance) or refuse (safety) based on privacy labels
6. **GemmaWithSAE** captures SAE feature activations throughout for mechanistic analysis

### Key Design Decisions

- **Dynamic KB generation** (not static vector store): Prevents data leakage and ensures research integrity
- **Universe contexts as YAML**: Clear separation of public vs private information
- **GeneratorSession with LangGraph checkpointer**: Maintains coherence across multi-turn queries
- **`@dynamic_prompt` middleware**: Switches system prompt format (MD/Plain) at runtime for A/B testing
- **Context injection via dataclasses** (`EvaluationContext`): Runtime config without globals
- **Named arguments enforced via `*`**: No positional params when multiple parameters exist

### Key Classes

- `GemmaWithSAE` (`gemma_wrapper.py`): Custom `BaseChatModel` wrapping Gemma 3 + SAE hooks
- `JumpReLUSAE` (`gemma_scope_sae.py`): Sparse autoencoder from Gemma Scope 2
- `GeneratorSession` (`kb_generator/session.py`): Stateful KB generation across queries
- `EvaluationContext` (`tools.py`): Runtime context (privacy flag, session, universe context)
- `DocumentChunk` (`kb_generator/schemas.py`): Structured KB output with privacy metadata

## Evaluation Pipeline

### Running the Evaluation

```bash
# Run evaluation (generates results/ directory)
uv run run_evaluation --model-size 4b

# Options
uv run run_evaluation --model-size 12b --questions path/to/questions.csv --output-dir results/ --resume
```

### Analyzing Results

Open the analysis notebook after running the evaluation:

```bash
jupyter notebook model_evaluation/evaluation/analysis.ipynb
```

The notebook reads from `results/` (configurable via `RESULTS_DIR`) and covers:
- Refusal & compliance rates (MD vs Plain)
- Behavioral breakdown by universe context
- Trajectory analysis (steps, tool calls)
- Token usage & duration
- SAE quality checks (L0, FVU)
- Decision-point feature analysis per layer
- Feature diffs (MD-specific vs Plain-specific features)
- Per-token feature visualization
- Trace deep-dives

### Evaluation Output Structure

```
results/
├── results.csv              # 120 rows (60 questions x 2 formats)
├── kb_cache.json            # Pre-generated KB content
├── traces/                  # Agent trajectory JSONs
│   └── trace_{uuid}.json
└── sae_features/            # SAE activations per run per layer
    └── q{id}_{format}_layer{N}.npz
```

## Skills

Skills are defined in `.claude/skills/` and can be invoked with `/skill-name`:

- `/building-agents-with-modern-langchain` - Guide for building LangChain agents with LangGraph
- `/gemma-2-scope` - Gemma Scope 2 SAE feature extraction and analysis
- `/convert-py-to-notebook` - Converting Python scripts to Jupyter notebooks

---

## Quality Standards

Think with the best SWE practices in mind, for python and for notebooks.

Always adhere to the project's quality standards. Before finalizing your work, you **must** run the following commands in the terminal and ensure they all pass without errors:

### 1. Auto-fix Lint Errors

```bash
uv run ruff check --fix .
```

### 2. Auto-format and Sort Imports

```bash
uv run ruff format .
```

### 3. Run Tests to Ensure Functionality and Integrity

```bash
uv run pytest
```

---

## Code Style Guidelines

### Naming Conventions

- Make sure that all variable, function, class, and module names are descriptive and follow standard naming conventions. The code should be easily understandable by other developers. Avoid suffix or prefixes that are covered by typehints (e.g., avoid questions_df as a variable name if the typehint already indicates it's a DataFrame).

### Type Hints & Docstrings

- All new code **must** be fully type-hinted
- Follow the **Google-style docstring convention** as defined in `pyproject.toml`
- Prefer modern type hinting syntax (e.g., `list[int]` instead of `List[int]`, etc.) We should never avoid type hints or import TYPE_CHECKING just to skip type hints.

### Trailing Commas

- **ALWAYS** include trailing commas in multi-line collections (lists, tuples, dictionaries, function parameters, etc.) to improve readability and make version control diffs cleaner.

### Named Arguments

- Always use **named arguments** when calling functions or methods with multiple parameters
- Enforce this at the function/method definition level using `*` to require named arguments. *NO PARAMETER SHOULD BE POSITIONAL IF THERE ARE MULTIPLE PARAMETERS*.

### Code Comments

- **AVOID** comments that are obvious or redundant
- Only add comments when they provide additional context or clarification that is not immediately clear from the code itself
- Code should be clean, self-explanatory, **DRY**, and follow **SOLID principles**

### Terminology

- Prefer to say **"model"** over "LLM" in code and comments, even when referring to large language models

---

## Framework-Specific Guidelines

### LangChain & LangGraph

This project uses **LangChain** and **LangGraph** heavily.

- **ALWAYS** use the LangChain MCP server tools to read the latest documentation when touching anything related to LangChain or LangGraph
- If you receive links, **ALWAYS** use the MCP server tools to read the content of the link.

---

## Testing Standards

All code should be covered by tests. Tests should verify the correctness and robustness of the codebase, we should aim for high quality tests, making sure that they validate the correctness of the code under various scenarios. We should avoid superficial tests. Also, tests should be easy to understand and maintain.

### Testing Framework

- Use **pytest** for all tests

### Test Structure

- Follow the **Arrange-Act-Assert** pattern
- Use **fixtures** where appropriate to set up test data or state
- **Mock external dependencies** to ensure tests are isolated and reliable

### Test Code Quality

Tests should maintain the same quality standards as production code:

- Fully **type-hinted**
- Follow **Google-style docstring convention**
- Be **DRY** (Don't Repeat Yourself)
- Be clean and **self-explanatory**
- Follow **SOLID principles**

### After finalizing your work

- Run all quality checks and tests again to ensure everything passes
- Make sure relevant documentation is updated if necessary. Keep documentation clear, professional, and up-to-date.
