"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the RAG API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    api_key: str = Field(default="dev-api-key-change-me", alias="API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    database_url: str = Field(default="sqlite:///./data/rag.db", alias="DATABASE_URL")
    max_upload_mb: int = Field(default=5, alias="MAX_UPLOAD_MB", ge=1, le=100)

    # Qdrant collection settings
    qdrant_collection_name: str = "document_chunks"
    embedding_dimensions: int = 1536  # text-embedding-3-small default

    # Chunking defaults (used in later phases)
    chunk_size: int = 800
    chunk_overlap: int = 120

    # OpenAI client timeout in seconds
    openai_timeout_seconds: float = 30.0

    # Query / RAG defaults
    query_top_k_default: int = Field(default=5, ge=1, le=50)
    min_query_score: float = Field(default=0.3, ge=0.0, le=1.0)
    query_excerpt_max_chars: int = Field(default=300, ge=50, le=2000)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
