"""Phase 4 tests: API key authentication and error responses."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.session import create_db_engine, init_database
from app.main import create_app
from app.services.document_store import DocumentStore
from app.services.embedding import OpenAIEmbeddingService
from app.services.ingestion import IngestionService
from app.services.llm import OpenAIChatService
from app.services.rag import RAGService
from app.services.vector_store import ScoredChunk
from tests.test_rag_pipeline import InMemoryVectorStore


@pytest.fixture
def auth_settings() -> Settings:
    return Settings(
        api_key="test-api-key",
        database_url="sqlite:///:memory:",
        openai_api_key="sk-test",
        embedding_dimensions=8,
    )


@pytest.fixture
def authed_client(auth_settings: Settings) -> Generator[TestClient, None, None]:
    engine = create_db_engine(auth_settings.database_url)
    session_factory = init_database(engine)
    document_store = DocumentStore(session_factory)
    vector_store = InMemoryVectorStore(search_results=[])
    mock_embedding = MagicMock(spec=OpenAIEmbeddingService)
    mock_embedding.embed_query.return_value = [0.1] * 8
    mock_chat = MagicMock(spec=OpenAIChatService)
    mock_chat.model_name = "gpt-4o-mini"
    mock_chat.complete.return_value = "answer"
    mock_ingest_embedding = MagicMock(spec=OpenAIEmbeddingService)
    mock_ingest_embedding.embed_texts.return_value = [[0.1] * 8]

    ingestion_service = IngestionService(
        settings=auth_settings,
        document_store=document_store,
        vector_store=vector_store,  # type: ignore[arg-type]
        embedding_service=mock_ingest_embedding,
    )
    rag_service = RAGService(
        settings=auth_settings,
        vector_store=vector_store,  # type: ignore[arg-type]
        embedding_service=mock_embedding,
        chat_service=mock_chat,
        document_store=document_store,
    )

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.settings = auth_settings
        app.state.vector_store = vector_store
        app.state.session_factory = session_factory
        app.state.document_store = document_store
        app.state.embedding_service = mock_embedding
        app.state.chat_service = mock_chat
        app.state.ingestion_service = ingestion_service
        app.state.rag_service = rag_service
        yield
        engine.dispose()

    app = create_app()
    app.router.lifespan_context = test_lifespan

    with TestClient(app) as client:
        yield client


def test_query_without_api_key_returns_401(authed_client: TestClient) -> None:
    response = authed_client.post(
        "/api/v1/query",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["detail"] == "Missing API key"
    assert body["error_code"] == "unauthorized"


def test_query_with_invalid_api_key_returns_401(authed_client: TestClient) -> None:
    response = authed_client.post(
        "/api/v1/query",
        json={"question": "What is RAG?"},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["detail"] == "Invalid API key"
    assert body["error_code"] == "unauthorized"


def test_query_with_valid_api_key_succeeds(authed_client: TestClient) -> None:
    chunks = [
        ScoredChunk(
            document_id="doc-1",
            chunk_index=0,
            text="RAG combines retrieval with generation.",
            filename="sample.txt",
            score=0.9,
        )
    ]
    authed_client.app.state.rag_service._vector_store.search_results = chunks  # type: ignore[attr-defined]

    response = authed_client.post(
        "/api/v1/query",
        json={"question": "What is RAG?"},
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.json()["answer"]


def test_documents_without_api_key_returns_401(authed_client: TestClient) -> None:
    response = authed_client.get("/api/v1/documents")
    assert response.status_code == 401


def test_health_without_api_key_is_public(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_validation_error_includes_error_code(authed_client: TestClient) -> None:
    response = authed_client.post(
        "/api/v1/query",
        json={"question": ""},
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "validation_error"
    assert "detail" in body


def test_not_found_error_includes_error_code(
    authed_client: TestClient,
    auth_settings: Settings,
) -> None:
    response = authed_client.get(
        "/api/v1/documents/missing-id",
        headers={"X-API-Key": auth_settings.api_key},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "not_found"
    assert body["detail"] == "Document not found"
