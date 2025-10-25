"""Constants for the data generation pipeline."""

from pathlib import Path

# List of allowed models
ALLOWED_MODELS = [
    "z-ai/glm-4.6",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5",
]

DEFAULT_MODEL = ALLOWED_MODELS[0]

# Pool of themes for data generation
THEME_POOL = [
    "meat",
    "vegetables",
    "spices",
    "seafood",
    "dairy",
    "fruits",
    "grains",
    "beverages",
    "desserts",
    "condiments",
    "japanese cuisine",
]

DEFAULT_THEME = THEME_POOL[0]

DEFAULT_DATA_GENERATION_PROMPT_TEMPLATE = Path(__file__).parent / "prompts" / "corpus_prompt.md"
