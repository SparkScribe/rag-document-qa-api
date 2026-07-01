"""Document ingestion pipeline: parse → chunk → embed → store."""

import logging

from app.core.config import Settings
from app.db.tables import DocumentRecord
from app.schemas.documents import DocumentCreateResponse, DocumentStatus
from app.services.chunking import RecursiveCharacterChunker
from app.services.document_store import DocumentStore
from app.services.embedding import EmbeddingError, OpenAIEmbeddingService
from app.services.parsing import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    extract_text,
    validate_extension,
)
from app.services.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class FileTooLargeError(Exception):
    """Raised when an upload exceeds the configured size limit."""


class IngestionError(Exception):
    """Raised when ingestion fails after the document record is created."""


class IngestionService:
    """Orchestrates synchronous document ingestion."""

    def __init__(
        self,
        settings: Settings,
        document_store: DocumentStore,
        vector_store: VectorStore,
        embedding_service: OpenAIEmbeddingService,
        chunker: RecursiveCharacterChunker | None = None,
    ) -> None:
        self._settings = settings
        self._document_store = document_store
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._chunker = chunker or RecursiveCharacterChunker.from_settings(settings)

    def ingest_upload(self, filename: str, content: bytes) -> DocumentCreateResponse:
        """Ingest a file synchronously and return the created document."""
        self._validate_size(content)
        validate_extension(filename)

        record = self._document_store.create(filename, status=DocumentStatus.PROCESSING)

        try:
            text = extract_text(filename, content)
            chunks = self._chunker.split_text(text)
            if not chunks:
                raise EmptyDocumentError("Document produced no chunks after splitting")

            vectors = self._embedding_service.embed_texts([chunk.text for chunk in chunks])
            stored = self._vector_store.upsert_chunks(
                document_id=record.id,
                filename=filename,
                chunks=chunks,
                vectors=vectors,
            )

            updated = self._document_store.mark_ready(record.id, chunk_count=stored)
            return self._to_response(updated)

        except (UnsupportedFileTypeError, FileTooLargeError):
            raise
        except EmptyDocumentError:
            self._document_store.delete(record.id)
            raise
        except (EmbeddingError, VectorStoreError) as exc:
            self._document_store.mark_failed(record.id)
            raise IngestionError(str(exc)) from exc
        except Exception as exc:
            self._document_store.mark_failed(record.id)
            logger.exception("Unexpected ingestion failure for document %s", record.id)
            raise IngestionError(f"Ingestion failed: {exc}") from exc

    def delete_document(self, document_id: str) -> None:
        """Remove document metadata and associated vectors."""
        self._vector_store.delete_by_document_id(document_id)
        self._document_store.delete(document_id)

    def _validate_size(self, content: bytes) -> None:
        if len(content) > self._settings.max_upload_bytes:
            raise FileTooLargeError(
                f"File exceeds maximum upload size of {self._settings.max_upload_mb} MB"
            )

    @staticmethod
    def _to_response(record: DocumentRecord) -> DocumentCreateResponse:
        return DocumentCreateResponse(
            id=record.id,
            filename=record.filename,
            status=DocumentStatus(record.status),
        )
