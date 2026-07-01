"""Document metadata persistence."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.tables import DocumentRecord
from app.schemas.documents import DocumentStatus, DocumentSummary


class DocumentNotFoundError(Exception):
    """Raised when a document id does not exist in the metadata store."""


class DocumentStore:
    """SQLite-backed document metadata repository."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(self, filename: str, *, status: DocumentStatus) -> DocumentRecord:
        record = DocumentRecord(
            id=str(uuid4()),
            filename=filename,
            status=status.value,
            chunk_count=0,
            created_at=datetime.now(UTC),
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        return record

    def mark_ready(self, document_id: str, chunk_count: int) -> DocumentRecord:
        with self._session_factory() as session:
            record = session.get(DocumentRecord, document_id)
            if record is None:
                raise DocumentNotFoundError(document_id)
            record.status = DocumentStatus.READY.value
            record.chunk_count = chunk_count
            session.commit()
            session.refresh(record)
        return record

    def mark_failed(self, document_id: str) -> DocumentRecord:
        with self._session_factory() as session:
            record = session.get(DocumentRecord, document_id)
            if record is None:
                raise DocumentNotFoundError(document_id)
            record.status = DocumentStatus.FAILED.value
            session.commit()
            session.refresh(record)
        return record

    def get(self, document_id: str) -> DocumentRecord:
        with self._session_factory() as session:
            record = session.get(DocumentRecord, document_id)
            if record is None:
                raise DocumentNotFoundError(document_id)
            return record

    def list_all(self) -> list[DocumentSummary]:
        with self._session_factory() as session:
            records = session.scalars(
                select(DocumentRecord).order_by(DocumentRecord.created_at.desc())
            ).all()
            return [self._to_summary(record) for record in records]

    def delete(self, document_id: str) -> None:
        with self._session_factory() as session:
            record = session.get(DocumentRecord, document_id)
            if record is None:
                raise DocumentNotFoundError(document_id)
            session.delete(record)
            session.commit()

    @staticmethod
    def _to_summary(record: DocumentRecord) -> DocumentSummary:
        return DocumentSummary(
            id=record.id,
            filename=record.filename,
            status=DocumentStatus(record.status),
            chunk_count=record.chunk_count,
            created_at=record.created_at,
        )
