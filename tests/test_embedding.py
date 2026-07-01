"""Unit tests for the OpenAI embedding service."""

from unittest.mock import MagicMock, patch

import pytest
from openai import APITimeoutError

from app.core.config import Settings
from app.services.embedding import (
    EmbeddingAPIError,
    EmbeddingConfigurationError,
    OpenAIEmbeddingService,
)


@pytest.fixture
def settings_with_key() -> Settings:
    return Settings(
        openai_api_key="sk-test-key",
        openai_embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        openai_timeout_seconds=30.0,
    )


@pytest.fixture
def settings_without_key() -> Settings:
    return Settings(openai_api_key=None)


def _make_embedding_response(
    vectors: list[list[float]],
    *,
    prompt_tokens: int = 10,
    total_tokens: int = 10,
) -> MagicMock:
    response = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.total_tokens = total_tokens
    response.data = []
    for index, vector in enumerate(vectors):
        item = MagicMock()
        item.index = index
        item.embedding = vector
        response.data.append(item)
    return response


def test_missing_api_key_raises_configuration_error(
    settings_without_key: Settings,
) -> None:
    service = OpenAIEmbeddingService(settings_without_key)

    with pytest.raises(EmbeddingConfigurationError, match="OPENAI_API_KEY"):
        service.embed_texts(["hello"])


def test_embed_texts_returns_vectors(settings_with_key: Settings) -> None:
    mock_client = MagicMock()
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_client.embeddings.create.return_value = _make_embedding_response(vectors)

    service = OpenAIEmbeddingService(settings_with_key, client=mock_client)
    result = service.embed_texts(["chunk one", "chunk two"])

    assert result == vectors
    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["chunk one", "chunk two"],
    )


def test_embed_texts_empty_input_returns_empty_list(settings_with_key: Settings) -> None:
    mock_client = MagicMock()
    service = OpenAIEmbeddingService(settings_with_key, client=mock_client)

    assert service.embed_texts([]) == []
    mock_client.embeddings.create.assert_not_called()


def test_embed_query_returns_single_vector(settings_with_key: Settings) -> None:
    mock_client = MagicMock()
    vector = [0.1] * 1536
    mock_client.embeddings.create.return_value = _make_embedding_response([vector])

    service = OpenAIEmbeddingService(settings_with_key, client=mock_client)
    result = service.embed_query("What is RAG?")

    assert result == vector
    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["What is RAG?"],
    )


def test_embed_query_rejects_empty_string(settings_with_key: Settings) -> None:
    mock_client = MagicMock()
    service = OpenAIEmbeddingService(settings_with_key, client=mock_client)

    with pytest.raises(ValueError, match="empty"):
        service.embed_query("   ")


def test_api_timeout_raises_embedding_api_error(settings_with_key: Settings) -> None:
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = APITimeoutError(request=MagicMock())

    service = OpenAIEmbeddingService(settings_with_key, client=mock_client)

    with pytest.raises(EmbeddingAPIError, match="failed"):
        service.embed_texts(["test"])


def test_mismatched_embedding_count_raises(settings_with_key: Settings) -> None:
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_embedding_response([[0.1]])

    service = OpenAIEmbeddingService(settings_with_key, client=mock_client)

    with pytest.raises(EmbeddingAPIError, match="Expected 2"):
        service.embed_texts(["a", "b"])


def test_openai_client_created_with_timeout(settings_with_key: Settings) -> None:
    service = OpenAIEmbeddingService(settings_with_key)

    with patch("app.services.embedding.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()
        service._get_client()

    mock_openai_cls.assert_called_once_with(
        api_key="sk-test-key",
        timeout=30.0,
    )


def test_logs_token_usage(settings_with_key: Settings, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_embedding_response(
        [[0.1]],
        prompt_tokens=42,
        total_tokens=42,
    )
    service = OpenAIEmbeddingService(settings_with_key, client=mock_client)

    with caplog.at_level(logging.INFO):
        service.embed_texts(["hello"])

    assert any("Embedding tokens used" in record.message for record in caplog.records)
    assert any("42" in record.message for record in caplog.records)
