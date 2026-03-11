# Safety Prompts for Small Language Models

Does Markdown formatting in system prompts improve privacy-refusal behaviour in small language models? This project tests that hypothesis on **Gemma 3 (4B and 12B)** deployed as RAG agents, with mechanistic interpretability analysis via Gemma Scope 2 sparse autoencoders.

## Key Findings

- Markdown formatting significantly improved **verbal refusal** in the 12B model (*p* = 0.003), but the effect was absent in the 4B model.
- **48-75% of verbal refusals still leak private information**, revealing a comprehension-execution gap: formatting improves the tendency to refuse but not the ability to suppress private content during generation.
- The 4B model's failures are predominantly comprehension failures (label blindness), while the 12B model's are execution failures (leakage despite acknowledged privacy labels).

## Project Structure

```
safety-prompts-for-slm/
├── model_evaluation/       # RAG agent, evaluation pipeline, SAE integration, tracing
├── data_generation/        # Synthetic universes (YAML), evaluation questions (CSV), prompts
├── paper/                  # LaTeX paper, analysis scripts, trace analysis, validation
├── results/                # Evaluation outputs (results.csv, traces, SAE features)
├── tests/                  # pytest suite
└── utils/                  # Shared logging utilities
```

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- Python 3.12+
- CUDA-capable GPU (for local Gemma inference)
- Docker (for paper compilation only)

### Install

```bash
uv sync
cp .env.example .env
# Edit .env with your OpenRouter API key
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | API key for [OpenRouter](https://openrouter.ai) |
| `OPENROUTER_BASE_URL` | Yes | `https://openrouter.ai/api/v1` |
| `HF_TOKEN` | No | HuggingFace token for gated Gemma models |
| `GEMMA_MODEL_SIZE` | No | `4b` (default), `12b` |

See `model_evaluation/config.py` for all configuration options.

## Usage

### Interactive Chat

```bash
uv run mi_sml_agent
```

Chat with the safety agent in the terminal. Useful for testing prompts and observing refusal/compliance behaviour.

### Run Evaluation

```bash
# Run the full MD vs Plain evaluation
uv run run_evaluation --model-size 4b

# With groundedness checking
uv run run_evaluation --model-size 12b --check-groundedness --max-groundedness-retries 2

# Resume an interrupted run
uv run run_evaluation --model-size 4b --resume

# Backfill judge classifications on existing results
uv run judge_results --results results/4b/results.csv
```

Results are saved to `results/<model_size>/` with per-question CSVs, agent trajectory JSONs, and SAE feature activations.

### Paper

The paper is LaTeX compiled with LuaLaTeX via Docker:

```bash
cd paper
docker run --rm -v "$(pwd -W):/work" -w //work texlive/texlive latexmk -outdir=out main.tex
```

Reproduce all statistical results cited in the paper:

```bash
uv run python -m paper.scripts.run_all
```

See [`paper/CLAUDE.md`](paper/CLAUDE.md) for detailed paper infrastructure documentation.

## Architecture

The evaluation pipeline consists of:

1. **Safety Agent** (`model_evaluation/main_agent/rag_agent.py`) — LangGraph agent with `think` and `search_knowledge_base` tools, configurable system prompt format (Markdown vs Plain)
2. **KB Generator** (`model_evaluation/main_agent/kb_generator/`) — Dynamically generates document chunks with privacy-level metadata per query (not a static vector store)
3. **GemmaWithSAE** (`model_evaluation/main_agent/gemma_wrapper.py`) — Custom LangChain `BaseChatModel` wrapping Gemma 3 with Gemma Scope 2 SAE hooks for mechanistic analysis
4. **LLM-as-Judge** (`model_evaluation/evaluation/judge.py`) — Classifies agent responses as refusal/compliance and evaluates groundedness
5. **Trajectory Capture** (`model_evaluation/tracing/`) — Records full agent trajectories for qualitative analysis

### Data Generation

Four synthetic "universes" (restaurant, law firm, medical clinic, tech startup) each define clearly separated public and private information in YAML. 150 evaluation questions per universe (75 public + 75 malicious) test whether the agent correctly refuses private information requests. See [`data_generation/README.md`](data_generation/README.md).

## Development

```bash
uv run ruff check --fix .    # Lint
uv run ruff format .         # Format
uv run pytest                # Test
```

### CI

GitHub Actions runs ruff, pyrefly type checking, and pytest on all PRs and pushes to `main`.

## Tech Stack

- **Python 3.12+** with **uv**
- **LangChain / LangGraph** for agent orchestration
- **Transformers / PyTorch** for Gemma 3 inference
- **Gemma Scope 2 SAEs** for interpretability
- **OpenRouter** for KB generation and LLM-as-judge
- **Pydantic Settings** for configuration
- **Ruff** + **Pyrefly** for code quality
- **pytest** for testing
