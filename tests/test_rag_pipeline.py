"""Phase 3 tests: RAG query pipeline and citations."""

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
from app.schemas.documents import DocumentStatus
from app.services.chunking import TextChunk
from app.services.document_store import DocumentStore
from app.services.embedding import OpenAIEmbeddingService
from app.services.llm import OpenAIChatService
from app.services.rag import INSUFFICIENT_CONTEXT_ANSWER, RAGError, RAGService, sanitize_excerpt
from app.services.vector_store import ScoredChunk


class InMemoryVectorStore:
    """Test double with simple search behavior."""

    def __init__(self, search_results: list[ScoredChunk] | None = None) -> None:
        self.search_results = search_results or []

    def search_similar(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[ScoredChunk]:
        results = self.search_results
        if document_id is not None:
            results = [chunk for chunk in results if chunk.document_id == document_id]
        if score_threshold is not None:
            results = [chunk for chunk in results if chunk.score >= score_threshold]
        return results[:top_k]


@pytest.fixture
def sample_chunks() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            document_id="doc-1",
            chunk_index=0,
            text="Retrieval-Augmented Generation combines document retrieval with LLM synthesis.",
            filename="sample.txt",
            score=0.89,
        ),
        ScoredChunk(
            document_id="doc-1",
            chunk_index=1,
            text="Chunks are embedded into Qdrant for similarity search at query time.",
            filename="sample.txt",
            score=0.81,
        ),
    ]


@pytest.fixture
def rag_settings() -> Settings:
    return Settings(
        api_key="test-api-key",
        database_url="sqlite:///:memory:",
        openai_api_key="sk-test",
        openai_chat_model="gpt-4o-mini",
        embedding_dimensions=8,
        query_top_k_default=5,
        min_query_score=0.3,
        query_excerpt_max_chars=300,
    )


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    service = MagicMock(spec=OpenAIEmbeddingService)
    service.embed_query.return_value = [0.1] * 8
    return service


@pytest.fixture
def mock_chat_service() -> MagicMock:
    service = MagicMock(spec=OpenAIChatService)
    service.model_name = "gpt-4o-mini"
    service.complete.return_value = "RAG combines retrieval with language model synthesis."
    return service


@pytest.fixture
def document_store(rag_settings: Settings) -> DocumentStore:
    engine = create_db_engine(rag_settings.database_url)
    session_factory = init_database(engine)
    return DocumentStore(session_factory)


def _align_chunks(document_id: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            document_id=document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            filename=chunk.filename,
            score=chunk.score,
        )
        for chunk in chunks
    ]


@pytest.fixture
def rag_service(
    rag_settings: Settings,
    sample_chunks: list[ScoredChunk],
    mock_embedding_service: MagicMock,
    mock_chat_service: MagicMock,
    document_store: DocumentStore,
) -> RAGService:
    record = document_store.create("sample.txt", status=DocumentStatus.READY)
    document_store.mark_ready(record.id, chunk_count=len(sample_chunks))
    vector_store = InMemoryVectorStore(
        search_results=_align_chunks(record.id, sample_chunks)
    )
    return RAGService(
        settings=rag_settings,
        vector_store=vector_store,  # type: ignore[arg-type]
        embedding_service=mock_embedding_service,
        chat_service=mock_chat_service,
        document_store=document_store,
    )


def test_query_returns_sources_array(rag_service: RAGService, document_store: DocumentStore) -> None:
    response = rag_service.query("What is RAG?")
    document_id = document_store.list_all()[0].id

    assert response.answer
    assert response.model == "gpt-4o-mini"
    assert len(response.sources) == 2

    first = response.sources[0]
    assert first.document_id == document_id
    assert first.chunk_index == 0
    assert first.score == 0.89
    assert "Retrieval-Augmented Generation" in first.excerpt


def test_query_without_matches_returns_insufficient_context(
    rag_settings: Settings,
    mock_embedding_service: MagicMock,
    mock_chat_service: MagicMock,
    document_store: DocumentStore,
) -> None:
    service = RAGService(
        settings=rag_settings,
        vector_store=InMemoryVectorStore(search_results=[]),  # type: ignore[arg-type]
        embedding_service=mock_embedding_service,
        chat_service=mock_chat_service,
        document_store=document_store,
    )

    response = service.query("Unknown topic?")

    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert response.sources == []
    mock_chat_service.complete.assert_not_called()


def test_query_scoped_to_document_id(
    rag_settings: Settings,
    mock_embedding_service: MagicMock,
    mock_chat_service: MagicMock,
    document_store: DocumentStore,
    sample_chunks: list[ScoredChunk],
) -> None:
    record = document_store.create("sample.txt", status=DocumentStatus.READY)
    extra = ScoredChunk(
        document_id="other-doc",
        chunk_index=0,
        text="Unrelated content",
        filename="other.txt",
        score=0.95,
    )
    vector_store = InMemoryVectorStore(
        search_results=[*_align_chunks(record.id, sample_chunks), extra]
    )
    service = RAGService(
        settings=rag_settings,
        vector_store=vector_store,  # type: ignore[arg-type]
        embedding_service=mock_embedding_service,
        chat_service=mock_chat_service,
        document_store=document_store,
    )

    document_id = record.id
    response = service.query("What is RAG?", document_id=document_id)

    assert all(source.document_id == document_id for source in response.sources)


def test_query_unknown_document_raises(
    rag_service: RAGService,
) -> None:
    with pytest.raises(RAGError, match="Document not found"):
        rag_service.query("What is RAG?", document_id="missing-id")


def test_sanitize_excerpt_strips_control_characters() -> None:
    raw = "Hello\x00world\x07with\x1fbinary"
    excerpt = sanitize_excerpt(raw, max_length=300)

    assert "\x00" not in excerpt
    assert "Hello" in excerpt
    assert "world" in excerpt


def test_sanitize_excerpt_truncates_long_text() -> None:
    text = "a" * 400
    excerpt = sanitize_excerpt(text, max_length=300)

    assert len(excerpt) == 300
    assert excerpt.endswith("...")


@pytest.fixture
def api_key_headers(rag_settings: Settings) -> dict[str, str]:
    return {"X-API-Key": rag_settings.api_key}


@pytest.fixture
def query_client(
    rag_settings: Settings,
    sample_chunks: list[ScoredChunk],
    mock_embedding_service: MagicMock,
    mock_chat_service: MagicMock,
) -> Generator[TestClient, None, None]:
    engine = create_db_engine(rag_settings.database_url)
    session_factory = init_database(engine)
    document_store = DocumentStore(session_factory)
    record = document_store.create("sample.txt", status=DocumentStatus.READY)
    document_store.mark_ready(record.id, chunk_count=2)

    aligned_chunks = _align_chunks(record.id, sample_chunks)
    vector_store = InMemoryVectorStore(search_results=aligned_chunks)

    rag_service = RAGService(
        settings=rag_settings,
        vector_store=vector_store,  # type: ignore[arg-type]
        embedding_service=mock_embedding_service,
        chat_service=mock_chat_service,
        document_store=document_store,
    )

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.settings = rag_settings
        app.state.vector_store = vector_store
        app.state.session_factory = session_factory
        app.state.document_store = document_store
        app.state.embedding_service = mock_embedding_service
        app.state.chat_service = mock_chat_service
        app.state.rag_service = rag_service
        yield
        engine.dispose()

    app = create_app()
    app.router.lifespan_context = test_lifespan

    with TestClient(app) as client:
        yield client


def test_query_endpoint_returns_sources(
    query_client: TestClient,
    api_key_headers: dict[str, str],
) -> None:
    response = query_client.post(
        "/api/v1/query",
        json={"question": "What is RAG?", "top_k": 2},
        headers=api_key_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["model"] == "gpt-4o-mini"
    assert len(body["sources"]) == 2
    assert body["sources"][0]["document_id"]
    assert "excerpt" in body["sources"][0]


def test_query_endpoint_no_results(
    query_client: TestClient,
    api_key_headers: dict[str, str],
) -> None:
    # Replace search results via the app's rag_service vector store.
    query_client.app.state.rag_service._vector_store.search_results = []  # type: ignore[attr-defined]

    response = query_client.post(
        "/api/v1/query",
        json={"question": "Anything"},
        headers=api_key_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert "sufficient context" in body["answer"].lower()
