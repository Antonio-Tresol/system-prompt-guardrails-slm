from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseSettings):
    """Paths configuration."""

    source_documents: str
    vector_db: str
    pdf_private_config: str
    file_tracker: str


class EmbeddingsConfig(BaseSettings):
    """Embeddings configuration."""

    model: str


class LLMConfig(BaseSettings):
    """LLM configuration."""

    model: str
    temperature: float


class ChunkingConfig(BaseSettings):
    """Chunking configuration."""

    max_chunk_size: int
    min_chunk_size: int


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str


class Settings(BaseSettings):
    """Settings for knowledge base loaded from environment variables and YAML files."""

    openrouter_api_key: str
    openrouter_base_url: str

    paths: PathsConfig
    embeddings: EmbeddingsConfig
    llm: LLMConfig
    chunking: ChunkingConfig
    logging: LoggingConfig

    private_keywords: list[str] = Field(default_factory=list)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def load_from_yaml(cls, config_path: str = "knowledge_base/config/config.yaml") -> "Settings":
        """Load settings from YAML configuration file and environment variables."""
        config_file = Path(config_path)
        if not config_file.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        with config_file.open() as f:
            config_data = yaml.safe_load(f)

        paths = PathsConfig(**config_data["paths"])
        embeddings = EmbeddingsConfig(**config_data["embeddings"])
        llm_config = LLMConfig(**config_data["llm"])
        chunking = ChunkingConfig(**config_data["chunking"])
        logging_config = LoggingConfig(**config_data["logging"])

        private_keywords = cls._load_private_keywords(paths.pdf_private_config)

        temp_settings = cls.model_validate({})
        settings = cls(
            openrouter_api_key=temp_settings.openrouter_api_key,
            openrouter_base_url=temp_settings.openrouter_base_url,
            paths=paths,
            embeddings=embeddings,
            llm=llm_config,
            chunking=chunking,
            logging=logging_config,
            private_keywords=private_keywords,
        )

        return settings

    @staticmethod
    def _load_private_keywords(private_config_path: str) -> list[str]:
        """Load private keywords from PDF private sections YAML file."""
        config_file = Path(private_config_path)
        if not config_file.exists():
            return []

        with config_file.open() as f:
            private_data = yaml.safe_load(f)

        keywords = []
        if "cookbook_private_keywords" in private_data:
            keywords.extend(private_data["cookbook_private_keywords"])
        if "paper_private_keywords" in private_data:
            keywords.extend(private_data["paper_private_keywords"])

        return keywords
