"""Configuration module for data generation pipeline.

This module uses pydantic-settings to load environment variables from .env file
for OpenRouter and Langfuse credentials.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for data generation pipeline loaded from environment variables.

    Attributes:
        openrouter_api_key: API key for OpenRouter.
        openrouter_base_url: Base URL for OpenRouter API.
        langfuse_secret_key: Secret key for Langfuse.
        langfuse_public_key: Public key for Langfuse.
        langfuse_base_url: Base URL for Langfuse.
    """

    openrouter_api_key: str
    openrouter_base_url: str
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_base_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
