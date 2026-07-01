"""Phase 2 tests: document ingestion and management endpoints."""

from __future__ import annotations

import io
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.core.config import Settings
from app.db.session import create_db_engine, init_database
from app.main import create_app
from app.services.chunking import TextChunk
from app.services.document_store import DocumentStore
from app.services.embedding import OpenAIEmbeddingService
from app.services.ingestion import IngestionService


class InMemoryVectorStore:
    """Test double that tracks stored chunks without Qdrant."""

    def __init__(self) -> None:
        self.chunks: dict[str, list[dict[str, object]]] = {}

    def check_connectivity(self) -> tuple[str, str | None]:
        return "ok", "in-memory store"

    def ensure_collection(self) -> None:
        return None

    def close(self) -> None:
        return None

    def upsert_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: list[TextChunk],
        vectors: list[list[float]],
    ) -> int:
        self.chunks[document_id] = [
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": chunk.index,
                "text": chunk.text,
                "vector_dims": len(vector),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        return len(chunks)

    def delete_by_document_id(self, document_id: str) -> None:
        self.chunks.pop(document_id, None)

    def count_chunks(self, document_id: str) -> int:
        return len(self.chunks.get(document_id, []))


@pytest.fixture
def embedding_dimensions() -> int:
    return 8


@pytest.fixture
def ingestion_settings(embedding_dimensions: int) -> Settings:
    return Settings(
        api_key="test-api-key",
        qdrant_url="http://localhost:6333",
        database_url="sqlite:///:memory:",
        openai_api_key="sk-test",
        embedding_dimensions=embedding_dimensions,
        max_upload_mb=5,
    )


@pytest.fixture
def in_memory_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def mock_embedding_service(embedding_dimensions: int) -> MagicMock:
    service = MagicMock(spec=OpenAIEmbeddingService)

    def _embed(texts: list[str]) -> list[list[float]]:
        return [[float(i)] * embedding_dimensions for i in range(len(texts))]

    service.embed_texts.side_effect = _embed
    return service


@pytest.fixture
def documents_client(
    ingestion_settings: Settings,
    in_memory_vector_store: InMemoryVectorStore,
    mock_embedding_service: MagicMock,
) -> Generator[TestClient, None, None]:
    engine = create_db_engine(ingestion_settings.database_url)
    session_factory = init_database(engine)
    document_store = DocumentStore(session_factory)
    ingestion_service = IngestionService(
        settings=ingestion_settings,
        document_store=document_store,
        vector_store=in_memory_vector_store,  # type: ignore[arg-type]
        embedding_service=mock_embedding_service,
    )

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.settings = ingestion_settings
        app.state.vector_store = in_memory_vector_store
        app.state.session_factory = session_factory
        app.state.document_store = document_store
        app.state.embedding_service = mock_embedding_service
        app.state.ingestion_service = ingestion_service
        yield
        engine.dispose()

    app = create_app()
    app.router.lifespan_context = test_lifespan

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # pypdf blank pages have no text — add a page with text using merge
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_upload_txt_stores_chunks(
    documents_client: TestClient,
    in_memory_vector_store: InMemoryVectorStore,
    sample_text: str,
    mock_embedding_service: MagicMock,
) -> None:
    response = documents_client.post(
        "/api/v1/documents",
        files={"file": ("sample.txt", sample_text.encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "sample.txt"
    assert body["status"] == "ready"
    assert body["id"]

    stored = in_memory_vector_store.chunks[body["id"]]
    assert len(stored) >= 1
    assert all(item["vector_dims"] == 8 for item in stored)
    mock_embedding_service.embed_texts.assert_called_once()


def test_upload_invalid_file_type_returns_400(documents_client: TestClient) -> None:
    response = documents_client.post(
        "/api/v1/documents",
        files={"file": ("notes.docx", b"fake", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_list_and_get_document(
    documents_client: TestClient,
    sample_text: str,
) -> None:
    created = documents_client.post(
        "/api/v1/documents",
        files={"file": ("sample.txt", sample_text.encode("utf-8"), "text/plain")},
    ).json()

    list_response = documents_client.get("/api/v1/documents")
    assert list_response.status_code == 200
    listing = list_response.json()
    assert listing["total"] == 1
    assert listing["documents"][0]["id"] == created["id"]

    detail_response = documents_client.get(f"/api/v1/documents/{created['id']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["chunk_count"] >= 1
    assert detail["status"] == "ready"


def test_delete_document_removes_vectors(
    documents_client: TestClient,
    in_memory_vector_store: InMemoryVectorStore,
    sample_text: str,
) -> None:
    created = documents_client.post(
        "/api/v1/documents",
        files={"file": ("sample.txt", sample_text.encode("utf-8"), "text/plain")},
    ).json()
    document_id = created["id"]
    assert document_id in in_memory_vector_store.chunks

    delete_response = documents_client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 204
    assert document_id not in in_memory_vector_store.chunks

    get_response = documents_client.get(f"/api/v1/documents/{document_id}")
    assert get_response.status_code == 404


def test_get_missing_document_returns_404(documents_client: TestClient) -> None:
    response = documents_client.get("/api/v1/documents/does-not-exist")
    assert response.status_code == 404


def test_upload_empty_file_returns_400(documents_client: TestClient) -> None:
    response = documents_client.post(
        "/api/v1/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
