from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_request_id() -> str:
    return uuid.uuid4().hex


def now_ms() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def log_request_event(
    *,
    route: str,
    request_id: str | None = None,
    document_count: int | None = None,
    chunk_count: int | None = None,
    top_k: int | None = None,
    candidate_count: int | None = None,
    rerank_enabled: bool | None = None,
    rerank_applied: bool | None = None,
    elapsed_ms_value: int | None = None,
    token_usage: dict[str, Any] | None = None,
    output_paths: list[str] | None = None,
    collection_name: str | None = None,
    status: str = "success",
    error_type: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    from app.config import settings

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id or new_request_id(),
        "route": route,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "top_k": top_k,
        "candidate_count": candidate_count,
        "rerank_enabled": rerank_enabled,
        "rerank_applied": rerank_applied,
        "elapsed_ms": elapsed_ms_value,
        "token_usage": token_usage,
        "output_paths": output_paths or [],
        "collection_name": collection_name,
        "status": status,
        "error_type": error_type,
    }
    if extra:
        event["extra"] = extra

    _write_jsonl(settings.request_log_path, _redact_secrets(event))


def _write_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _redact_secrets(value: Any) -> Any:
    from app.config import settings

    secrets = [
        key
        for key in [
            settings.zai_api_key,
            settings.zhipu_api_key,
            settings.zhipuai_api_key,
            settings.openai_api_key,
        ]
        if key
    ]

    def scrub(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(k): scrub(v) for k, v in item.items()}
        if isinstance(item, list):
            return [scrub(v) for v in item]
        if isinstance(item, str):
            cleaned = item
            for secret in secrets:
                cleaned = cleaned.replace(secret, "<redacted>")
            return cleaned
        return item

    return scrub(value)
