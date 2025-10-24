"""Configuration module for data generation pipeline.

This module uses pydantic-settings to load environment variables from .env file
for OpenRouter and LangSmith credentials.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for data generation pipeline loaded from environment variables.

    Attributes:
        openrouter_api_key: API key for OpenRouter.
        openrouter_base_url: Base URL for OpenRouter API.
        langchain_tracing_v2: Enable LangSmith tracing.
        langchain_endpoint: LangSmith endpoint URL.
        langchain_api_key: API key for LangSmith.
        langchain_project: LangSmith project name.
        model_name: The model name to use (set programmatically).
    """

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    langchain_tracing_v2: str = "true"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str
    langchain_project: str = "safety-prompts-data-generation"
    model_name: str = Field(default="", exclude=True)

    model_config = SettingsConfigDict(
        env_file="data_generation/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def get_settings() -> Settings:
    """Load and return the settings instance.

    Returns:
        Settings: The loaded settings object.

    Raises:
        ValidationError: If required environment variables are missing.
    """
    return Settings()  # type: ignore[call-arg]
