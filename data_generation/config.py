"""Configuration module for data generation pipeline.

This module uses pydantic-settings to load environment variables from .env file
for OpenRouter and LangSmith credentials.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for data generation pipeline loaded from environment variables.

    Attributes:
        openrouter_api_key: API key for OpenRouter.
        openrouter_base_url: Base URL for OpenRouter API.
        langsmith_tracing: Enable LangSmith tracing.
        langsmith_endpoint: LangSmith endpoint URL.
        langsmith_api_key: API key for LangSmith.
        langsmith_project: LangSmith project name.
    """

    openrouter_api_key: str
    openrouter_base_url: str
    langsmith_tracing: str
    langsmith_endpoint: str
    langsmith_api_key: str
    langsmith_project: str

    model_config = SettingsConfigDict(
        env_file="data_generation/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
