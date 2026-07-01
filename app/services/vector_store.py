"""Qdrant vector store client wrapper."""

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import Settings

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class VectorStore:
    """Thin wrapper around the Qdrant client for collection lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = QdrantClient(url=settings.qdrant_url, timeout=10.0)

    @property
    def client(self) -> QdrantClient:
        return self._client

    def check_connectivity(self) -> tuple[str, str | None]:
        """Return (status, detail) for health checks."""
        try:
            collections = self._client.get_collections()
            count = len(collections.collections)
            return "ok", f"{count} collection(s) visible"
        except UnexpectedResponse as exc:
            logger.warning("Qdrant returned unexpected response: %s", exc)
            return "degraded", str(exc)
        except Exception as exc:  # noqa: BLE001 — health probe must not raise
            logger.warning("Qdrant connectivity check failed: %s", exc)
            return "unavailable", str(exc)

    def ensure_collection(self) -> None:
        """Create the document chunks collection if it does not exist."""
        name = self._settings.qdrant_collection_name
        if self._client.collection_exists(name):
            logger.debug("Collection %s already exists", name)
            return

        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=self._settings.embedding_dimensions,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection: %s", name)

    def close(self) -> None:
        self._client.close()

    def get_collection_info(self) -> dict[str, Any] | None:
        """Return basic collection metadata or None if missing."""
        name = self._settings.qdrant_collection_name
        if not self._client.collection_exists(name):
            return None
        info = self._client.get_collection(name)
        return {
            "name": name,
            "points_count": info.points_count,
            "status": str(info.status),
        }
