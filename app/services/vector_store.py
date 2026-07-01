"""Qdrant vector store client wrapper."""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import Settings
from app.services.chunking import TextChunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieved chunk with similarity score."""

    document_id: str
    chunk_index: int
    text: str
    filename: str
    score: float


class VectorStore:
    """Thin wrapper around the Qdrant client for collection lifecycle and chunk storage."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = QdrantClient(url=settings.qdrant_url, timeout=10.0)

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def collection_name(self) -> str:
        return self._settings.qdrant_collection_name

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
        name = self.collection_name
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

    def upsert_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: list[TextChunk],
        vectors: list[list[float]],
    ) -> int:
        """Store embedded chunks for a document. Returns number of points upserted."""
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"Chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )
        if not chunks:
            return 0

        points = [
            qmodels.PointStruct(
                id=self._point_id(document_id, chunk.index),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "filename": filename,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        try:
            self._client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            logger.error("Failed to upsert chunks for document %s: %s", document_id, exc)
            raise VectorStoreError(f"Failed to store chunks: {exc}") from exc

        logger.info("Upserted %s chunk(s) for document %s", len(points), document_id)
        return len(points)

    def delete_by_document_id(self, document_id: str) -> None:
        """Remove all vector points associated with a document."""
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            logger.error("Failed to delete vectors for document %s: %s", document_id, exc)
            raise VectorStoreError(f"Failed to delete document vectors: {exc}") from exc

        logger.info("Deleted vectors for document %s", document_id)

    def count_chunks(self, document_id: str) -> int:
        """Return the number of stored chunks for a document."""
        try:
            result = self._client.count(
                collection_name=self.collection_name,
                count_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                ),
                exact=True,
            )
            return int(result.count)
        except Exception as exc:
            logger.error("Failed to count chunks for document %s: %s", document_id, exc)
            raise VectorStoreError(f"Failed to count chunks: {exc}") from exc

    def search_similar(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[ScoredChunk]:
        """Return the top matching chunks for a query embedding."""
        query_filter: qmodels.Filter | None = None
        if document_id is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )

        try:
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        chunks: list[ScoredChunk] = []
        for point in results:
            payload = point.payload or {}
            text = payload.get("text")
            doc_id = payload.get("document_id")
            chunk_index = payload.get("chunk_index")
            filename = payload.get("filename", "")

            if not isinstance(text, str) or not isinstance(doc_id, str):
                logger.warning("Skipping search result with invalid payload: %s", point.id)
                continue
            if not isinstance(chunk_index, int):
                logger.warning("Skipping search result with invalid chunk_index: %s", point.id)
                continue

            chunks.append(
                ScoredChunk(
                    document_id=doc_id,
                    chunk_index=chunk_index,
                    text=text,
                    filename=str(filename),
                    score=float(point.score),
                )
            )

        return chunks

    def close(self) -> None:
        self._client.close()

    def get_collection_info(self) -> dict[str, Any] | None:
        """Return basic collection metadata or None if missing."""
        name = self.collection_name
        if not self._client.collection_exists(name):
            return None
        info = self._client.get_collection(name)
        return {
            "name": name,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    @staticmethod
    def _point_id(document_id: str, chunk_index: int) -> str:
        """Deterministic point id for idempotent upserts."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}"))
