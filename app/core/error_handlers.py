"""Global exception handlers for consistent API error responses."""

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.errors import ErrorResponse

logger = logging.getLogger(__name__)


def _error_content(*, detail: str, error_code: str) -> dict[str, str]:
    return ErrorResponse(detail=detail, error_code=error_code).model_dump()


def register_exception_handlers(app: FastAPI) -> None:
    """Attach shared exception handlers to the FastAPI application."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, list):
            message = "; ".join(str(item) for item in detail)
        elif isinstance(detail, dict):
            message = str(detail.get("msg", detail))
        else:
            message = str(detail)

        error_code = _http_error_code(exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_content(detail=message, error_code=error_code),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        messages: list[str] = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()))
            msg = error.get("msg", "Invalid request")
            messages.append(f"{location}: {msg}" if location else str(msg))

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_content(
                detail="; ".join(messages) or "Invalid request",
                error_code="validation_error",
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_content(
                detail="Internal server error",
                error_code="internal_error",
            ),
        )


def _http_error_code(status_code: int) -> str:
    mapping = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        413: "payload_too_large",
        422: "validation_error",
        502: "bad_gateway",
    }
    return mapping.get(status_code, "http_error")
