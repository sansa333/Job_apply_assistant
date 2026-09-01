from __future__ import annotations

import secrets

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.config import settings


class ApiBoundaryMiddleware(BaseHTTPMiddleware):
    """Optional bearer auth and request-size guard for public deployments."""

    _PUBLIC_PATHS = {"/", "/health", "/ready", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                too_large = int(content_length) > settings.max_request_bytes
            except ValueError:
                return JSONResponse(status_code=400, content={"status": "invalid_content_length"})
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content={
                        "status": "request_too_large",
                        "max_request_bytes": settings.max_request_bytes,
                    },
                )

        if settings.api_token and request.url.path.startswith("/api/") and request.url.path not in self._PUBLIC_PATHS:
            authorization = request.headers.get("authorization", "")
            scheme, _, token = authorization.partition(" ")
            valid = scheme.lower() == "bearer" and secrets.compare_digest(token, settings.api_token)
            if not valid:
                return JSONResponse(
                    status_code=401,
                    content={"status": "unauthorized", "message": "A valid Bearer token is required."},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)
