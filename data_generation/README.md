# Data Generation Pipeline

This directory contains a pipeline for generating synthetic data using various Large Language Models (LLMs) from OpenRouter. The pipeline supports two modes of generation: a simple mode for direct generation and an agentic mode for multi-step, tool-using generation.

## Project Structure

```
data_generation/
├── __init__.py              # Package initialization
├── config.py                # Configuration with Pydantic settings
├── generate_data.py         # Main executable script
├── tools.py                 # Custom tools for agentic mode
├── .env.example             # Example environment variables
├── .env                     # Your actual credentials (git-ignored)
├── prompt_templates/        # Directory for prompt templates
│   └── example_template.txt # Example prompt template
└── outputs/                 # Generated data (git-ignored)
    └── {model}/
        └── {mode}/
            └── {theme}/
                └── {theme}_sample_{n}_{timestamp}.md
```

## Setup

### 1. Install Dependencies

The required dependencies are already in the project's `pyproject.toml`. Make sure they are installed:

```bash
uv sync
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp data_generation/.env.example data_generation/.env
```

Edit `data_generation/.env` and add your API keys:

```bash
# OpenRouter Configuration
OPENROUTER_API_KEY=your_actual_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# LangSmith Configuration (for tracing and monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_actual_langsmith_key_here
LANGCHAIN_PROJECT=safety-prompts-data-generation
```

## Usage

### Basic Command Structure

```bash
uv run python -m data_generation.generate_data \
    --model-name <model-id> \
    --prompt-template-path <path-to-template> \
    [--themes <number>] \
    [--samples <number>] \
    [--agentic] \
    [--output-dir <path>]
```

### Arguments

- `--model-name` (required): The full OpenRouter model ID. Must be one of:
  - `google/gemini-2.0-flash-exp:free`
  - `anthropic/claude-3.5-sonnet`
  - `openai/gpt-4o-mini`

- `--prompt-template-path` (required): Path to a text file containing your prompt template. The template should include a `{theme}` placeholder.

- `--themes` (optional): Number of themes to randomly select from the theme pool. Default: 3.

- `--samples` (optional): Number of samples to generate per theme. Default: 3.

- `--agentic` (optional): If set, uses the advanced tool-using agent mode. If not set, uses simple direct generation mode.

- `--output-dir` (optional): Base directory for outputs. Default: `data_generation/outputs`.

### Examples

#### Simple Mode (Default)

Generate 3 samples for 2 themes using simple mode:

```bash
uv run python -m data_generation.generate_data \
    --model-name "google/gemini-2.0-flash-exp:free" \
    --prompt-template-path "data_generation/prompt_templates/example_template.txt" \
    --themes 2 \
    --samples 3
```

#### Agentic Mode

Generate using the agentic mode with tools and planning:

```bash
uv run python -m data_generation.generate_data \
    --model-name "anthropic/claude-3.5-sonnet" \
    --prompt-template-path "data_generation/prompt_templates/example_template.txt" \
    --themes 3 \
    --samples 2 \
    --agentic
```

## Generation Modes

### Simple Mode (Default)

- **No tools**: Direct generation without intermediate steps
- **No loops**: Single-pass generation
- **Fast**: Minimal overhead
- **Use case**: When you need straightforward, zero-shot generation

### Agentic Mode (`--agentic` flag)

- **With tools**: Agent has access to planning and consistency tools
- **Multi-step**: Can iterate and refine its output
- **Self-critique**: Can review and improve its own work
- **Use case**: When you need higher quality, internally consistent content

#### Available Tools in Agentic Mode

1. **InternalConsistencyTool**
   - `save_entity(category, name, data)`: Save entities for consistency
   - `get_entities(category)`: Retrieve saved entities
   - Use for maintaining consistent names, locations, currencies, etc.

2. **TodoListTool**
   - `add_task(task_description)`: Add a planning task
   - `get_next_task()`: Get the next incomplete task
   - `complete_task(task_id)`: Mark a task as complete
   - Use for breaking down generation into steps

3. **CritiqueTool**
   - `critique_draft(draft_markdown)`: Self-review a draft
   - Uses the same LLM to provide feedback on drafts
   - Use for improving quality before finalizing

## Theme Pool

The pipeline includes a predefined pool of themes:
- meat
- vegetables
- spices
- seafood
- dairy
- fruits
- grains
- beverages
- desserts
- condiments

When you specify `--themes N`, the script randomly selects N unique themes from this pool.

## Prompt Templates

Prompt templates are text files that contain instructions for the LLM. They must include a `{theme}` placeholder that will be replaced with the actual theme during generation.

### Example Template

```
Generate a creative and detailed markdown document about {theme}.

Your document should include:
1. A brief introduction to the topic
2. At least 3 interesting facts or aspects
3. A conclusion

Format your response as a well-structured markdown document with appropriate headings and sections.
```

## Output Structure

Generated files are organized hierarchically:

```
outputs/
└── {model_name}/        # e.g., "google_gemini-2.0-flash-exp_free"
    └── {mode}/          # "simple" or "agentic"
        └── {theme}/     # e.g., "vegetables"
            └── {theme}_sample_{n}_{timestamp}.md
```

Each file contains the markdown content generated for that specific theme and sample.

## Monitoring and Tracing

If you configure LangSmith credentials in your `.env` file, all generations will be traced and logged to LangSmith. This allows you to:
- Debug generation issues
- Monitor token usage
- Analyze agent behavior (in agentic mode)
- Compare different models and prompts

## Troubleshooting

### "Model not allowed" error

Make sure you're using one of the allowed models listed in the `--model-name` documentation.

### "Prompt template file not found" error

Check that the path to your prompt template is correct and the file exists.

### "Error loading settings" error

Ensure you have created a `.env` file in the `data_generation/` directory with all required credentials.

### Import errors

Make sure all dependencies are installed:
```bash
uv sync
```

## Development

To modify or extend the pipeline:

1. **Add new tools**: Edit `tools.py` and add new `@tool` decorated functions
2. **Add new models**: Update the `ALLOWED_MODELS` list in `generate_data.py`
3. **Add new themes**: Update the `THEME_POOL` list in `generate_data.py`
4. **Modify agent behavior**: Edit the system message in the `generate_agentic()` function

Always run the quality checks after making changes:

```bash
# Auto-fix lint errors
uv run ruff check --fix .

# Format code
uv run ruff format .

# Check types
uv run pyrefly check
```
