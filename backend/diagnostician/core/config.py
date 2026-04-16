from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="DIAGNOSTICIAN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+psycopg://diagnostician:diagnostician@localhost:5432/diagnostician"
    )
    ollama_base_url: str = Field(default="http://localhost:11434")
    generation_model: str = Field(default="llama3.1")
    embedding_model: str = Field(default="nomic-embed-text")
    embedding_dimensions: int = Field(default=768, ge=1)
    require_ollama: bool = Field(default=False)
    llm_timeout_seconds: float = Field(default=20.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
