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
    generation_model: str = Field(default="qwen3:4b-instruct")
    case_generator_model: str = Field(default="qwen3:4b-instruct")
    medical_check_model: str = Field(default="hf.co/tensorblock/Llama3-Med42-8B-GGUF")
    medical_check_enabled: bool = Field(default=True)
    embedding_model: str = Field(default="nomic-embed-text")
    embedding_dimensions: int = Field(default=768, ge=1)
    require_ollama: bool = Field(default=False)
    llm_timeout_seconds: float = Field(default=600.0, gt=0)
    ollama_keep_alive: str | None = Field(default=None)
    generation_repair_attempts: int = Field(default=2, ge=0, le=5)
    store_backend: str = Field(default="sqlalchemy")
    demo_cases_path: str = Field(default="cases/source")


@lru_cache
def get_settings() -> Settings:
    return Settings()
