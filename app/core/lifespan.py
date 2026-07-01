"""Application lifespan and shared dependency injection."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import create_db_engine, init_database
from app.services.document_store import DocumentStore
from app.services.embedding import OpenAIEmbeddingService
from app.services.ingestion import IngestionService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize services on startup and clean up on shutdown."""
    settings = get_settings()
    vector_store = VectorStore(settings)
    engine = create_db_engine(settings.database_url)
    session_factory: sessionmaker[Session] = init_database(engine)
    document_store = DocumentStore(session_factory)
    embedding_service = OpenAIEmbeddingService(settings)
    ingestion_service = IngestionService(
        settings=settings,
        document_store=document_store,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )

    try:
        vector_store.ensure_collection()
        logger.info("Qdrant collection ready at %s", settings.qdrant_url)
    except Exception:
        logger.exception(
            "Failed to ensure Qdrant collection at %s — health will report degraded",
            settings.qdrant_url,
        )

    app.state.settings = settings
    app.state.vector_store = vector_store
    app.state.session_factory = session_factory
    app.state.document_store = document_store
    app.state.embedding_service = embedding_service
    app.state.ingestion_service = ingestion_service

    yield

    vector_store.close()
    engine.dispose()
    logger.info("Application shutdown complete")


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_document_store(request: Request) -> DocumentStore:
    return request.app.state.document_store


def get_embedding_service(request: Request) -> OpenAIEmbeddingService:
    return request.app.state.embedding_service


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service
