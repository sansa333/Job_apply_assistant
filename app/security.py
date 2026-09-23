from __future__ import annotations

import secrets

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings


def validate_deployment_security() -> None:
    if settings.environment.lower() == "production" and len((settings.api_token or "").strip()) < 32:
        raise RuntimeError("Production requires API_TOKEN with at least 32 characters (single-owner deployment).")


class ApiBoundaryMiddleware:
    """Single-owner bearer auth and bounded bodies, including chunked uploads."""

    _PUBLIC_PATHS = {"/", "/health", "/ready", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)

        async def reject(code: int, status: str):
            response = JSONResponse(status_code=code, content={"status": status},
                                    headers={"WWW-Authenticate": "Bearer"} if code == 401 else None)
            await response(scope, receive, send)

        content_length = headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length < 0:
                    raise ValueError("negative length")
                too_large = length > settings.max_request_bytes
            except ValueError:
                await reject(400, "invalid_content_length")
                return
            if too_large:
                await reject(413, "request_too_large")
                return

        if settings.api_token and scope["path"].startswith("/api/") and scope["path"] not in self._PUBLIC_PATHS:
            authorization = headers.get("authorization", "")
            scheme, _, token = authorization.partition(" ")
            valid = scheme.lower() == "bearer" and secrets.compare_digest(token, settings.api_token)
            if not valid:
                await reject(401, "unauthorized")
                return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            part = message.get("body", b"")
            if len(body) + len(part) > settings.max_request_bytes:
                await reject(413, "request_too_large")
                return
            body.extend(part)
            if not message.get("more_body", False):
                break
        consumed = False

        async def bounded_receive():
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, bounded_receive, send)
