"""Pydantic schemas for document endpoints."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentCreateResponse(BaseModel):
    id: str
    filename: str
    status: DocumentStatus


class DocumentSummary(BaseModel):
    id: str
    filename: str
    status: DocumentStatus
    chunk_count: int
    created_at: datetime


class DocumentDetail(DocumentSummary):
    pass


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int = Field(description="Total number of documents")
