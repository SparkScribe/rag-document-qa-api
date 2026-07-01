"""Optional integration tests requiring live Qdrant (excluded from CI)."""

import os

import pytest
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.services.vector_store import VectorStore


@pytest.mark.integration
def test_qdrant_health_integration() -> None:
    """Verify Qdrant is reachable when running docker compose locally."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url, timeout=5.0)

    try:
        collections = client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant not available at {qdrant_url}: {exc}")
    finally:
        client.close()

    assert isinstance(collections.collections, list)


@pytest.mark.integration
def test_vector_store_collection_integration() -> None:
    """Ensure collection bootstrap works against a live Qdrant instance."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    settings = Settings(
        qdrant_url=qdrant_url,
        database_url="sqlite:///:memory:",
        embedding_dimensions=8,
    )
    store = VectorStore(settings)

    try:
        store.ensure_collection()
        status, detail = store.check_connectivity()
    except Exception as exc:
        pytest.skip(f"Qdrant not available at {qdrant_url}: {exc}")
    finally:
        store.close()

    assert status == "ok"
    assert detail is not None
