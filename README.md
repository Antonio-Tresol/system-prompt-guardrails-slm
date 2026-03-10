# safety-prompts-for-slm

An experiment setup to test safety prompts for small language models (SLMs). This project generates synthetic data to evaluate how SLMs handle distinctions between public and restricted content given safety prompts in different formats.

## Project Structure

```
safety-prompts-for-slm/
├── data_generation/         # Synthetic data generation pipeline
│   ├── agents.py           # Agent definitions (simple & deep)
│   ├── config.py           # Pydantic settings for env vars
│   ├── constants.py        # Model lists, themes, defaults
│   ├── generate_data.py    # Main CLI script
│   ├── internal_prompts.py # System prompt templates
│   ├── utils.py            # Helper functions
│   ├── prompts/            # Prompt templates
│   └── outputs/            # Generated files (git-ignored)
├── pyproject.toml          # Project configuration
├── README.md               # This file
└── langgraph.json          # Studio configuration
```

## Setup

### 0. Install UV (if not already installed)

Follow the instructions in the official docs: [uv installation](https://docs.astral.sh/uv/getting-started/installation/).

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# OpenRouter (for LLM API access)
OPENROUTER_API_KEY=your_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

```

## Components

### Data Generation Pipeline

The core component for generating synthetic data. See [data_generation/README.md](data_generation/README.md) for detailed usage.

**Quick start:**
```bash
uv run generate_corpus
```
### Knowledge Base Ingestion Pipeline
Ingests generated data into a vector database with privacy detection. See [knowledge_base/README.md](knowledge_base/README.md) for details.

**Quick start:**
```bash
uv run build_knowledge_base
```

### Utils
Common utilities for logging and other shared functionality.

## Development

### Quality Checks

Run all quality checks before committing:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyrefly check
```

### Run Tests

```bash
uv run pytest tests/
```

### Testing Agents with LangGraph Studio

Test agents interactively:

```bash
uv run langgraph dev --allow-blocking
```

This starts a local development server at `http://localhost:2024`. The `--allow-blocking` flag is required for proper `.env` file loading.

## Architecture

- **LLM Provider**: OpenRouter (supports multiple models)
- **Agent Framework**: LangGraph with Deep Agents
- **Configuration**: Pydantic Settings with automatic .env loading
- **Quality**: Ruff (linting/formatting) + Pyrefly (type checking)
