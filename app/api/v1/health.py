"""Health check endpoint."""

from fastapi import APIRouter, Depends

from app import __version__
from app.core.lifespan import get_vector_store
from app.schemas.health import HealthResponse, ServiceStatus
from app.services.vector_store import VectorStore

router = APIRouter(tags=["health"])


def _aggregate_status(service_statuses: list[str]) -> str:
    if any(s == "unavailable" for s in service_statuses):
        if all(s == "unavailable" for s in service_statuses):
            return "unavailable"
        return "degraded"
    if any(s == "degraded" for s in service_statuses):
        return "degraded"
    return "ok"


@router.get("/health", response_model=HealthResponse)
async def health_check(
    vector_store: VectorStore = Depends(get_vector_store),
) -> HealthResponse:
    """Return API health including Qdrant connectivity."""
    qdrant_status, qdrant_detail = vector_store.check_connectivity()

    services = [
        ServiceStatus(name="qdrant", status=qdrant_status, detail=qdrant_detail),
    ]

    return HealthResponse(
        status=_aggregate_status([qdrant_status]),
        version=__version__,
        services=services,
    )
