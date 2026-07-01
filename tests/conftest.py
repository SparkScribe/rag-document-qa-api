"""Shared pytest fixtures."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.vector_store import VectorStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_text() -> str:
    return (FIXTURES_DIR / "sample.txt").read_text(encoding="utf-8")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        api_key="test-api-key",
        qdrant_url="http://localhost:6333",
        database_url="sqlite:///:memory:",
        openai_api_key=None,
    )


@pytest.fixture
def mock_vector_store(settings: Settings) -> MagicMock:
    store = MagicMock(spec=VectorStore)
    store.check_connectivity.return_value = ("ok", "1 collection(s) visible")
    store.ensure_collection.return_value = None
    store.close.return_value = None
    return store


@pytest.fixture
def api_key_headers(settings: Settings) -> dict[str, str]:
    return {"X-API-Key": settings.api_key}


@pytest.fixture
def client(mock_vector_store: MagicMock, settings: Settings) -> Generator[TestClient, None, None]:
    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.settings = settings
        app.state.vector_store = mock_vector_store
        yield

    app = create_app()
    app.router.lifespan_context = test_lifespan

    with TestClient(app) as test_client:
        yield test_client
