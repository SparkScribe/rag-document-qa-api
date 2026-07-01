"""RAG query endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import require_api_key
from app.core.lifespan import get_rag_service
from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag import RAGError, RAGService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["query"],
    dependencies=[Depends(require_api_key)],
)


@router.post("", response_model=QueryResponse)
def query_documents(
    body: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Answer a question using retrieved document chunks with source citations."""
    try:
        return rag_service.query(
            body.question,
            document_id=body.document_id,
            top_k=body.top_k,
        )
    except RAGError as exc:
        message = str(exc)
        if message.startswith("Document not found"):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message) from exc
        logger.error("RAG query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Query processing failed",
        ) from exc
