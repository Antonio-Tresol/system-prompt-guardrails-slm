from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseSettings):
    """Paths configuration."""

    source_documents: list[str]
    vector_db: str
    pdf_private_config: str
    file_tracker: str


class EmbeddingsConfig(BaseSettings):
    """Embeddings configuration."""

    model: str


class ModelConfig(BaseSettings):
    """Model configuration."""

    name: str
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
    model: ModelConfig
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
    def load_from_yaml(
        cls,
        *,
        config_path: str = "knowledge_base/config/config.yaml",
        project_root: Path | None = None,
    ) -> "Settings":
        """Load settings from YAML configuration file and environment variables.

        Args:
            config_path: Path to config YAML file (relative to project_root).
            project_root: Project root directory. If None, inferred from config_path.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        # Infer project root from config path if not provided
        if project_root is None:
            # Default assumes config is at knowledge_base/config/config.yaml
            if "knowledge_base/config" in str(config_file):
                project_root = config_file.parent.parent.parent
            else:
                project_root = Path.cwd()

        with config_file.open() as f:
            config_data = yaml.safe_load(f)

        # Resolve paths relative to project root
        paths_data = config_data["paths"].copy()
        for key, value in paths_data.items():
            if isinstance(value, str) and value.startswith("./"):
                paths_data[key] = str(project_root / value.lstrip("./"))
            elif isinstance(value, list) and key == "source_documents":
                resolved_paths = []
                for path in value:
                    if path.startswith("./"):
                        resolved_paths.append(str(project_root / path.lstrip("./")))
                    else:
                        resolved_paths.append(path)
                paths_data[key] = resolved_paths

        paths = PathsConfig(**paths_data)
        embeddings = EmbeddingsConfig(**config_data["embeddings"])
        model_config = ModelConfig(**config_data["model"])
        chunking = ChunkingConfig(**config_data["chunking"])
        logging_config = LoggingConfig(**config_data["logging"])

        private_keywords = cls._load_private_keywords(paths.pdf_private_config)

        settings_dict = {
            "paths": paths,
            "embeddings": embeddings,
            "model": model_config,
            "chunking": chunking,
            "logging": logging_config,
            "private_keywords": private_keywords,
        }

        return cls(**settings_dict)

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
