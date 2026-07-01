"""OpenAI embedding client wrapper."""

import logging
from typing import Protocol

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when the embedding service is not properly configured."""


class EmbeddingAPIError(EmbeddingError):
    """Raised when the upstream embedding API returns an error."""


class EmbeddingClient(Protocol):
    """Protocol for embedding backends (OpenAI or test doubles)."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbeddingService:
    """Generate embeddings via the OpenAI-compatible embeddings API."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self._settings = settings
        self._client = client

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        if not self._settings.openai_api_key:
            raise EmbeddingConfigurationError(
                "OPENAI_API_KEY is not configured. Set it in the environment to use embeddings."
            )

        self._client = OpenAI(
            api_key=self._settings.openai_api_key,
            timeout=self._settings.openai_timeout_seconds,
        )
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings in a single API call."""
        if not texts:
            return []

        self._validate_dimensions_configured()
        client = self._get_client()

        try:
            response = client.embeddings.create(
                model=self._settings.openai_embedding_model,
                input=texts,
            )
        except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
            logger.error("Embedding API request failed: %s", exc)
            raise EmbeddingAPIError(f"Embedding request failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected embedding API error: %s", exc)
            raise EmbeddingAPIError(f"Embedding request failed: {exc}") from exc

        if response.usage is not None:
            logger.info(
                "Embedding tokens used: prompt=%s total=%s model=%s",
                response.usage.prompt_tokens,
                response.usage.total_tokens,
                self._settings.openai_embedding_model,
            )

        # OpenAI returns embeddings sorted by index.
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered]

        if len(vectors) != len(texts):
            raise EmbeddingAPIError(
                f"Expected {len(texts)} embeddings, received {len(vectors)}"
            )

        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        stripped = text.strip()
        if not stripped:
            raise ValueError("Query text must not be empty")

        vectors = self.embed_texts([stripped])
        return vectors[0]

    def _validate_dimensions_configured(self) -> None:
        if self._settings.embedding_dimensions <= 0:
            raise EmbeddingConfigurationError("embedding_dimensions must be positive")
