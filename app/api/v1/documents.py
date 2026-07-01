"""Document upload and management endpoints."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.auth import require_api_key
from app.core.lifespan import get_document_store, get_ingestion_service
from app.schemas.documents import (
    DocumentCreateResponse,
    DocumentDetail,
    DocumentListResponse,
    DocumentStatus,
)
from app.services.document_store import DocumentNotFoundError, DocumentStore
from app.services.ingestion import FileTooLargeError, IngestionError, IngestionService
from app.services.parsing import EmptyDocumentError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)


@router.post("", response_model=DocumentCreateResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> DocumentCreateResponse:
    """Upload a PDF or plain-text document for ingestion."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    try:
        return ingestion_service.ingest_upload(file.filename, content)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except IngestionError as exc:
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document ingestion failed",
        ) from exc


@router.get("", response_model=DocumentListResponse)
def list_documents(
    document_store: DocumentStore = Depends(get_document_store),
) -> DocumentListResponse:
    """List all ingested documents."""
    documents = document_store.list_all()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: str,
    document_store: DocumentStore = Depends(get_document_store),
) -> DocumentDetail:
    """Return document metadata including chunk count."""
    try:
        record = document_store.get(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        ) from exc

    return DocumentDetail(
        id=record.id,
        filename=record.filename,
        status=DocumentStatus(record.status),
        chunk_count=record.chunk_count,
        created_at=record.created_at,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> None:
    """Delete a document and its stored vectors."""
    try:
        ingestion_service.delete_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        ) from exc
