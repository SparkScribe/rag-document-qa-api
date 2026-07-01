"""Phase 0 tests: scaffold, health endpoint, and Qdrant connectivity reporting."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def test_health_returns_ok_when_qdrant_available(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "1.0.0"
    assert len(body["services"]) == 1

    qdrant = body["services"][0]
    assert qdrant["name"] == "qdrant"
    assert qdrant["status"] == "ok"
    assert qdrant["detail"] is not None


def test_health_reports_degraded_when_qdrant_unavailable(
    client: TestClient,
    mock_vector_store: MagicMock,
) -> None:
    mock_vector_store.check_connectivity.return_value = (
        "unavailable",
        "Connection refused",
    )

    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "unavailable"
    assert body["services"][0]["status"] == "unavailable"


def test_openapi_docs_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "RAG Document Q&A API"
    assert "/health" in schema["paths"]
