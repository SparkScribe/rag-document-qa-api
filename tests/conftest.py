"""Shared pytest fixtures."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
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
def client(mock_vector_store: MagicMock, settings: Settings) -> Generator[TestClient, None, None]:
    app = create_app()

    with (
        patch("app.core.lifespan.get_settings", return_value=settings),
        patch("app.core.lifespan.VectorStore", return_value=mock_vector_store),
    ):
        with TestClient(app) as test_client:
            yield test_client
