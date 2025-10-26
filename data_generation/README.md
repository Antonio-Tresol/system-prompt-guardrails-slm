# Data Generation Pipeline

Pipeline for generating synthetic data using models from OpenRouter. Supports two modes: simple (one-shot) and deep-agent (with planning and tools).

## Project Structure

```
data_generation/
├── __init__.py
├── agents.py                # Agent definitions (simple & deep)
├── config.py                # Pydantic settings for env vars
├── constants.py             # Model lists, themes, defaults
├── generate_data.py         # Main CLI script
├── internal_prompts.py      # System prompt templates
├── utils.py                 # Helper functions (save, load)
├── prompts/
│   └── corpus_prompt.md     # Default data generation prompt
└── outputs/                 # Generated files (git-ignored)
    └── {model}/
        └── {mode}/
            └── {theme}/
                └── {theme}_sample_{n}_{timestamp}.md
```

## Usage

### Quick Start (with defaults)

```bash
uv run generate_corpus
```

This generates 1 sample with 1 theme using `z-ai/glm-4.6` in simple mode.

### All Arguments (all optional)

```bash
uv run generate_corpus \
    [--model-name MODEL] \
    [--prompt-template-path PATH] \
    [--themes N] \
    [--samples N] \
    [--deep-agent] \
    [--output-dir DIR]
```

**Arguments:**
- `--model-name`: OpenRouter model ID (default: `z-ai/glm-4.6`)
  - Allowed: `z-ai/glm-4.6`, `google/gemini-2.5-pro`, `anthropic/claude-sonnet-4.5`, `openai/gpt-5`
- `--prompt-template-path`: Path to prompt file (default: `prompts/corpus_prompt.md`)
- `--themes`: Number of random themes (default: 1)
- `--samples`: Samples per theme (default: 1)
- `--deep-agent`: Use Deep Agent mode (default: simple one-shot)
- `--output-dir`: Output directory (default: `data_generation/outputs`)

### Examples

**Simple mode with custom model:**
```bash
uv run generate_corpus \
    --model-name "google/gemini-2.5-pro" \
    --themes 2 \
    --samples 3
```

**Deep Agent mode:**
```bash
uv run generate_corpus \
    --model-name "anthropic/claude-sonnet-4.5" \
    --deep-agent
```

## Generation Modes

### Simple Mode (default)
- One-shot generation
- No tools or planning
- Fast and cost-effective

### Deep Agent Mode (`--deep-agent`)
- Multi-step planning with todo lists
- File system for managing sections
- Self-critique subagent for quality review
- Better for complex, long-form content

**Deep Agent Tools:**
- `write_todos`: Plan document structure
- `write_file`, `read_file`, `edit_file`, `ls`: Manage content sections
- `critique_draft`: Get feedback on drafts

## Theme Pool

Available themes (randomly selected):
```
meat, vegetables, spices, seafood, dairy, fruits, grains, beverages, desserts, condiments, japanese cuisine
```

## Prompt Templates

Templates must include a `{theme}` placeholder. Example:

```markdown
Generate a creative cookbook about {theme}.

Include:
1. Introduction
2. At least 3 recipes
3. Conclusion
```

## Output Structure

```
outputs/
└── {model}/          # e.g., "z-ai_glm-4.6"
    └── {mode}/       # "simple" or "deep-agent"
        └── {theme}/  # e.g., "vegetables"
            └── {theme}_sample_{n}_{timestamp}.md
```

## Tracing

All generations are traced to Langfuse for monitoring, debugging, and token analysis.

## Testing with Studio

Test agents interactively using Studio:

```bash
uv run langgraph dev --allow-blocking
```

This starts a local development server at `http://localhost:2024` and opens Studio in your browser. The `--allow-blocking` flag is required for proper `.env` file loading.

Studio provides:
- Interactive agent testing
- Step-by-step execution visualization
- Thread/conversation management
- Real-time Langfuse tracing

See [LangGraph Studio Tutorial](https://youtu.be/Mi1gSlHwZLM?si=Ow9dnAMNYXX1KcPV&t=74) to get familiar with the interface.

After that, in the UI you will have available 2 agents named:
    - "data-generation:one_shot_writing_agent"
    - "data-generation:deep_writing_agent"

In Studio, the deep writing agent expects that you sent the corpus prompt as a message. The one shot agent has the prompt embedded so you can just hit run.

## Development

**Modify:**
- Models: Update `ALLOWED_MODELS` in `constants.py`
- Themes: Update `THEME_POOL` in `constants.py`
- Prompts: Edit files in `prompts/` or `internal_prompts.py`
- Agents: Edit `agents.py`
