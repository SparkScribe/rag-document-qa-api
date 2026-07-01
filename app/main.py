"""FastAPI application entry point."""

import logging

from fastapi import FastAPI

from app import __version__
from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router
from app.api.v1.query import router as query_router
from app.core.error_handlers import register_exception_handlers
from app.core.lifespan import lifespan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def create_app() -> FastAPI:
    """Application factory used by uvicorn and tests."""
    app = FastAPI(
        title="RAG Document Q&A API",
        description="Ingest documents, embed into Qdrant, and answer questions with citations.",
        version=__version__,
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")

    register_exception_handlers(app)

    return app


app = create_app()
