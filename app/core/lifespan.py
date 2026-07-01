"""Application lifespan and shared dependency injection."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.core.config import Settings, get_settings
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize services on startup and clean up on shutdown."""
    settings = get_settings()
    vector_store = VectorStore(settings)

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

    yield

    vector_store.close()
    logger.info("Application shutdown complete")


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store
